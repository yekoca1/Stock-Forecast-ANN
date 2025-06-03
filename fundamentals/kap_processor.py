import json
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import re
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KAPReportProcessor:
    """
    Core data pipeline for processing KAP reports and calculating stock price impacts
    """
    
    def __init__(self, data_directory: str = "."):
        """
        Initialize the processor
        
        Args:
            data_directory: Directory containing JSON files
        """
        self.data_directory = Path(data_directory)
        self.turkey_tz = pytz.timezone('Europe/Istanbul')
        self.processed_data = []
        
    def extract_dates_from_content(self, content: str) -> List[str]:
        """
        Extract potential dates from report content
        
        Args:
            content: Report content text
            
        Returns:
            List of found dates in DD.MM.YYYY format
        """
        # Pattern for DD.MM.YYYY format
        date_pattern = r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b'
        dates = re.findall(date_pattern, content)
        return dates
    
    def parse_json_report(self, file_path: str) -> Dict:
        """
        Parse a single JSON KAP report
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with extracted information
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            stock_code = data.get('stock_code', '').upper()
            content = data.get('news_content', '')
            
            # Extract dates from content
            found_dates = self.extract_dates_from_content(content)
            
            # Try to determine publication date
            publication_date = None
            if found_dates:
                # Use the most recent date as publication date
                try:
                    dates_parsed = [datetime.strptime(date, '%d.%m.%Y') for date in found_dates]
                    publication_date = max(dates_parsed)
                except:
                    logger.warning(f"Could not parse dates from {file_path}")
            
            return {
                'file_path': file_path,
                'stock_code': stock_code,
                'content': content,
                'publication_date': publication_date,
                'found_dates': found_dates,
                'raw_data': data
            }
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return None
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day (Monday-Friday)
        
        Args:
            date: Date to check
            
        Returns:
            True if trading day, False otherwise
        """
        return date.weekday() < 5  # Monday=0, Friday=4
    
    def get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day after given date
        
        Args:
            date: Starting date
            
        Returns:
            Next trading day
        """
        next_day = date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        return next_day
    
    def get_target_trading_date(self, publication_datetime: datetime, stock_code: str) -> datetime:
        """
        Determine which day's closing price to analyze based on publication time
        
        Args:
            publication_datetime: When the report was published
            stock_code: Stock symbol
            
        Returns:
            Target date for price analysis
        """
        # Convert to Turkey timezone
        if publication_datetime.tzinfo is None:
            pub_time_turkey = self.turkey_tz.localize(publication_datetime)
        else:
            pub_time_turkey = publication_datetime.astimezone(self.turkey_tz)
        
        # BIST trading hours: 10:00-18:00 Turkey Time
        trading_start = pub_time_turkey.replace(hour=10, minute=0, second=0, microsecond=0)
        trading_end = pub_time_turkey.replace(hour=18, minute=0, second=0, microsecond=0)
        
        # If published on weekend, target next trading day
        if not self.is_trading_day(pub_time_turkey):
            return self.get_next_trading_day(pub_time_turkey)
        
        # If published during trading hours, target same day
        if trading_start <= pub_time_turkey <= trading_end:
            return pub_time_turkey.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Published outside trading hours, target next trading day
            return self.get_next_trading_day(pub_time_turkey)
    
    def fetch_stock_prices(self, stock_code: str, target_date: datetime, days_before: int = 5) -> Dict:
        """
        Fetch stock prices around target date
        
        Args:
            stock_code: Stock symbol (will be converted to Yahoo Finance format)
            target_date: Target date for analysis
            days_before: Number of days before target to fetch
            
        Returns:
            Dictionary with price data
        """
        try:
            # Convert Turkish stock code to Yahoo Finance format
            # THYAO -> THYAO.IS (Istanbul Stock Exchange)
            if not stock_code.endswith('.IS'):
                yahoo_symbol = f"{stock_code}.IS"
            else:
                yahoo_symbol = stock_code
            
            # Calculate date range
            start_date = target_date - timedelta(days=days_before)
            end_date = target_date + timedelta(days=2)  # Buffer for weekends
            
            # Fetch data
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No price data found for {yahoo_symbol}")
                return None
            
            # Convert index to date strings for easier handling
            hist.index = hist.index.strftime('%Y-%m-%d')
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            # Find target date price
            target_price = None
            previous_price = None
            
            if target_date_str in hist.index:
                target_price = hist.loc[target_date_str, 'Close']
            
            # Find previous trading day price
            hist_dates = list(hist.index)
            if target_date_str in hist_dates:
                target_idx = hist_dates.index(target_date_str)
                if target_idx > 0:
                    previous_date = hist_dates[target_idx - 1]
                    previous_price = hist.loc[previous_date, 'Close']
            
            return {
                'yahoo_symbol': yahoo_symbol,
                'target_date': target_date_str,
                'target_price': target_price,
                'previous_price': previous_price,
                'price_data': hist.to_dict('index'),
                'data_available': target_price is not None and previous_price is not None
            }
            
        except Exception as e:
            logger.error(f"Error fetching prices for {stock_code}: {str(e)}")
            return None
    
    def calculate_price_change(self, price_data: Dict) -> Dict:
        """
        Calculate price change metrics
        
        Args:
            price_data: Dictionary with price information
            
        Returns:
            Dictionary with calculated metrics
        """
        if not price_data or not price_data.get('data_available'):
            return {
                'price_change_absolute': None,
                'price_change_percentage': None,
                'direction': None
            }
        
        target_price = price_data['target_price']
        previous_price = price_data['previous_price']
        
        price_change_abs = target_price - previous_price
        price_change_pct = (price_change_abs / previous_price) * 100
        direction = 1 if price_change_abs > 0 else (-1 if price_change_abs < 0 else 0)
        
        return {
            'price_change_absolute': price_change_abs,
            'price_change_percentage': price_change_pct,
            'direction': direction
        }
    
    def process_single_report(self, file_path: str) -> Dict:
        """
        Process a single KAP report end-to-end
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Processing {file_path}")
        
        # Parse JSON
        report_data = self.parse_json_report(file_path)
        if not report_data:
            return None
        
        if not report_data['stock_code'] or not report_data['publication_date']:
            logger.warning(f"Missing stock code or publication date in {file_path}")
            return report_data
        
        # Determine target trading date
        target_date = self.get_target_trading_date(
            report_data['publication_date'], 
            report_data['stock_code']
        )
        
        # Fetch stock prices
        price_data = self.fetch_stock_prices(report_data['stock_code'], target_date)
        
        # Calculate price changes
        price_metrics = self.calculate_price_change(price_data)
        
        # Combine all data
        result = {
            **report_data,
            'target_trading_date': target_date,
            'price_data': price_data,
            **price_metrics,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def process_directory(self) -> List[Dict]:
        """
        Process all JSON files in the directory
        
        Returns:
            List of processed reports
        """
        json_files = list(self.data_directory.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        results = []
        for file_path in json_files:
            result = self.process_single_report(str(file_path))
            if result:
                results.append(result)
        
        self.processed_data = results
        return results
    
    def save_processed_data(self, output_file: str = "processed_kap_data.json"):
        """
        Save processed data to JSON file
        
        Args:
            output_file: Output file name
        """
        if not self.processed_data:
            logger.warning("No processed data to save")
            return
        
        # Convert datetime objects to strings for JSON serialization
        serializable_data = []
        for item in self.processed_data:
            serializable_item = item.copy()
            for key, value in serializable_item.items():
                if isinstance(value, datetime):
                    serializable_item[key] = value.isoformat()
            serializable_data.append(serializable_item)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Processed data saved to {output_file}")
    
    def get_training_dataset(self) -> pd.DataFrame:
        """
        Convert processed data to pandas DataFrame for model training
        
        Returns:
            DataFrame ready for machine learning
        """
        if not self.processed_data:
            logger.warning("No processed data available")
            return pd.DataFrame()
        
        # Filter out records without price data
        valid_data = [
            item for item in self.processed_data 
            if item.get('price_change_percentage') is not None
        ]
        
        if not valid_data:
            logger.warning("No valid price data found")
            return pd.DataFrame()
        
        # Create DataFrame
        df_data = []
        for item in valid_data:
            row = {
                'file_path': item['file_path'],
                'stock_code': item['stock_code'],
                'publication_date': item['publication_date'].isoformat() if item['publication_date'] else None,
                'target_trading_date': item['target_trading_date'].isoformat() if item['target_trading_date'] else None,
                'content': item['content'],
                'content_length': len(item['content']),
                'price_change_percentage': item['price_change_percentage'],
                'price_change_absolute': item['price_change_absolute'],
                'direction': item['direction'],
                'target_price': item['price_data']['target_price'] if item['price_data'] else None,
                'previous_price': item['price_data']['previous_price'] if item['price_data'] else None,
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        logger.info(f"Created training dataset with {len(df)} records")
        
        return df


# Example usage and testing
def main():
    """
    Main function to demonstrate the pipeline
    """
    # Initialize processor
    processor = KAPReportProcessor(".")
    
    # Process all JSON files in current directory
    results = processor.process_directory()
    
    # Save processed data
    processor.save_processed_data("processed_kap_reports.json")
    
    # Get training dataset
    df = processor.get_training_dataset()
    
    if not df.empty:
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total reports processed: {len(results)}")
        print(f"Reports with valid price data: {len(df)}")
        print(f"Unique stocks: {df['stock_code'].nunique()}")
        
        print("\n=== PRICE CHANGE STATISTICS ===")
        print(f"Average price change: {df['price_change_percentage'].mean():.2f}%")
        print(f"Std deviation: {df['price_change_percentage'].std():.2f}%")
        print(f"Min change: {df['price_change_percentage'].min():.2f}%")
        print(f"Max change: {df['price_change_percentage'].max():.2f}%")
        
        print("\n=== DIRECTION DISTRIBUTION ===")
        direction_counts = df['direction'].value_counts()
        print(f"Positive impact: {direction_counts.get(1, 0)} reports")
        print(f"Negative impact: {direction_counts.get(-1, 0)} reports")
        print(f"No change: {direction_counts.get(0, 0)} reports")
        
        # Save training dataset
        df.to_csv("training_dataset.csv", index=False)
        print(f"\nTraining dataset saved to 'training_dataset.csv'")
    
    return processor, df


if __name__ == "__main__":
    processor, df = main()