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
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KAPReportProcessor:
    """
    Core data pipeline for processing KAP reports and calculating stock price impacts
    """
    
    def __init__(self, stock_code: str = None, project_directory: str = "."):
        """
        Initialize the processor
        
        Args:
            stock_code: Stock code to process (e.g., 'THYAO')
            project_directory: Project root directory containing notification folders
        """
        self.project_directory = Path(project_directory)
        self.stock_code = stock_code
        
        # Determine data directory
        if stock_code:
            self.data_directory = self.project_directory / f"{stock_code.lower()}_notifications"
        else:
            # Auto-detect notification folders
            self.data_directory = self._auto_detect_notification_folder()
            
        self.turkey_tz = pytz.timezone('Europe/Istanbul')
        self.processed_data = []
        
        logger.info(f"Initialized processor for directory: {self.data_directory}")
        
    def _auto_detect_notification_folder(self) -> Path:
        """
        Auto-detect notification folder if stock_code is not provided
        
        Returns:
            Path to the notification folder
        """
        # Look for folders ending with '_notifications'
        notification_folders = list(self.project_directory.glob("*_notifications"))
        
        if len(notification_folders) == 0:
            logger.error("No notification folders found!")
            logger.info("Available folders:")
            for item in self.project_directory.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item.name}")
            raise FileNotFoundError("No *_notifications folders found in project directory")
        elif len(notification_folders) == 1:
            folder = notification_folders[0]
            # Extract stock code from folder name
            self.stock_code = folder.name.replace("_notifications", "").upper()
            logger.info(f"Auto-detected stock code: {self.stock_code}")
            return folder
        else:
            logger.info("Multiple notification folders found:")
            for i, folder in enumerate(notification_folders):
                logger.info(f"  {i+1}. {folder.name}")
            raise ValueError("Multiple notification folders found. Please specify stock_code parameter.")
    
    def list_available_notification_folders(self) -> List[str]:
        """
        List all available notification folders
        
        Returns:
            List of folder names
        """
        notification_folders = list(self.project_directory.glob("*_notifications"))
        return [folder.name for folder in notification_folders]
        
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
    
    def parse_timestamp_field(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse the timestamp field from JSON
        
        Args:
            timestamp_str: Timestamp string from JSON
            
        Returns:
            Parsed datetime object or None
        """
        try:
            # Try different timestamp formats
            formats_to_try = [
                '%Y-%m-%d %H:%M:%S',  # 2025-06-04 11:13:24
                '%d.%m.%Y %H:%M:%S',  # 04.06.2025 11:13:24
                '%Y-%m-%d',           # 2025-06-04
                '%d.%m.%Y',          # 04.06.2025
            ]
            
            for fmt in formats_to_try:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return None
            
        except Exception as e:
            logger.warning(f"Error parsing timestamp {timestamp_str}: {str(e)}")
            return None
    
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
            timestamp_str = data.get('timestamp', '')
            
            # Extract dates from content for reference
            found_dates = self.extract_dates_from_content(content)
            
            # Priority 1: Use timestamp field if available
            publication_date = None
            if timestamp_str:
                publication_date = self.parse_timestamp_field(timestamp_str)
                if publication_date:
                    logger.info(f"Using timestamp field as publication date: {publication_date}")
            
            # Priority 2: Try to determine from content dates
            if publication_date is None and found_dates:
                try:
                    dates_parsed = [datetime.strptime(date, '%d.%m.%Y') for date in found_dates]
                    # Use the earliest date (most likely to be the actual event date)
                    publication_date = min(dates_parsed)
                    logger.info(f"Using content date as publication date: {publication_date}")
                except:
                    logger.warning(f"Could not parse dates from content in {file_path}")
            
            # Priority 3: Use file modification time as fallback
            if publication_date is None:
                try:
                    file_stat = os.stat(file_path)
                    publication_date = datetime.fromtimestamp(file_stat.st_mtime)
                    logger.info(f"Using file modification time as publication date: {publication_date}")
                except:
                    logger.warning(f"Could not determine publication date for {file_path}")
            
            return {
                'file_path': file_path,
                'stock_code': stock_code,
                'content': content,
                'publication_date': publication_date,
                'timestamp_field': timestamp_str,
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
    
    def get_previous_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before given date
        
        Args:
            date: Starting date
            
        Returns:
            Previous trading day
        """
        prev_day = date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        return prev_day
    
    def get_target_trading_date(self, publication_datetime: datetime, stock_code: str) -> datetime:
        """
        Determine which day's closing price to analyze based on publication time
        FIXED: Proper timezone handling
        
        Args:
            publication_datetime: When the report was published
            stock_code: Stock symbol
            
        Returns:
            Target date for price analysis (timezone-aware)
        """
        # Convert to Turkey timezone if not already timezone-aware
        if publication_datetime.tzinfo is None:
            pub_time_turkey = self.turkey_tz.localize(publication_datetime)
        else:
            pub_time_turkey = publication_datetime.astimezone(self.turkey_tz)
        
        # BIST trading hours: 10:00-18:00 Turkey Time
        # Create timezone-aware datetime objects for comparison
        trading_start = pub_time_turkey.replace(hour=10, minute=0, second=0, microsecond=0)
        trading_end = pub_time_turkey.replace(hour=18, minute=0, second=0, microsecond=0)
        
        # If published on weekend, target next trading day
        if not self.is_trading_day(pub_time_turkey):
            next_trading = self.get_next_trading_day(pub_time_turkey.replace(tzinfo=None))
            return self.turkey_tz.localize(next_trading.replace(hour=0, minute=0, second=0, microsecond=0))
        
        # If published during trading hours, target same day
        if trading_start <= pub_time_turkey <= trading_end:
            return pub_time_turkey.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Published outside trading hours, target next trading day
            next_trading = self.get_next_trading_day(pub_time_turkey.replace(tzinfo=None))
            return self.turkey_tz.localize(next_trading.replace(hour=0, minute=0, second=0, microsecond=0))
    
    def get_alternative_yahoo_symbols(self, stock_code: str) -> List[str]:
        """
        Get alternative Yahoo Finance symbols to try for Turkish stocks
        ENHANCED: Added more Turkish stock exchange variations
        
        Args:
            stock_code: Original stock code
            
        Returns:
            List of possible Yahoo Finance symbols
        """
        symbols_to_try = []
        
        # Turkish stock exchange variations
        if not stock_code.endswith('.IS'):
            symbols_to_try.extend([
                f"{stock_code}.IS",     # Istanbul Stock Exchange (most common)
                f"{stock_code}.IST",    # Alternative Istanbul format
                f"{stock_code}.XIST",   # Extended Istanbul format
                stock_code,             # Plain symbol
            ])
        else:
            symbols_to_try.append(stock_code)
        
        return symbols_to_try
    
    def fetch_stock_prices(self, stock_code: str, target_date: datetime, days_before: int = 15) -> Dict:
        """
        Fetch stock prices around target date with multiple symbol attempts
        FIXED: Better error handling and date management
        
        Args:
            stock_code: Stock symbol
            target_date: Target date for analysis (timezone-aware)
            days_before: Number of days before target to fetch
            
        Returns:
            Dictionary with price data
        """
        symbols_to_try = self.get_alternative_yahoo_symbols(stock_code)
        
        # Convert target_date to naive datetime for Yahoo Finance
        if target_date.tzinfo is not None:
            target_date_naive = target_date.replace(tzinfo=None)
        else:
            target_date_naive = target_date
        
        for yahoo_symbol in symbols_to_try:
            try:
                logger.info(f"Trying symbol: {yahoo_symbol}")
                
                # Calculate date range - use a more conservative approach
                # Ensure we're not requesting future dates
                end_date = min(target_date_naive + timedelta(days=5), datetime.now())
                start_date = target_date_naive - timedelta(days=days_before)
                
                logger.info(f"Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                
                # Fetch data with error handling
                ticker = yf.Ticker(yahoo_symbol)
                
                # Try to get basic info first to check if symbol exists
                try:
                    info = ticker.info
                    if not info or 'symbol' not in info:
                        logger.warning(f"Symbol {yahoo_symbol} not found or invalid")
                        continue
                except:
                    logger.warning(f"Could not get info for {yahoo_symbol}")
                    continue
                
                # Fetch historical data
                hist = ticker.history(start=start_date, end=end_date, auto_adjust=True, prepost=True)
                
                if hist.empty:
                    logger.warning(f"No price data found for {yahoo_symbol} in date range")
                    continue
                
                logger.info(f"Successfully fetched {len(hist)} records for {yahoo_symbol}")
                
                # Convert index to date strings for easier handling
                hist.index = hist.index.strftime('%Y-%m-%d')
                target_date_str = target_date_naive.strftime('%Y-%m-%d')
                
                # Find target date price or closest available date
                target_price = None
                previous_price = None
                actual_target_date = None
                actual_previous_date = None
                
                hist_dates = sorted(list(hist.index))
                
                # Find the closest trading date on or after target date
                for date_str in hist_dates:
                    if date_str >= target_date_str:
                        target_price = hist.loc[date_str, 'Close']
                        actual_target_date = date_str
                        logger.info(f"Found target price on: {date_str}")
                        break
                
                # If no date found on or after target, use the last available date
                if target_price is None and hist_dates:
                    actual_target_date = hist_dates[-1]
                    target_price = hist.loc[actual_target_date, 'Close']
                    logger.info(f"Using last available date: {actual_target_date}")
                
                # Find previous trading day price
                if actual_target_date and actual_target_date in hist_dates:
                    target_idx = hist_dates.index(actual_target_date)
                    if target_idx > 0:
                        actual_previous_date = hist_dates[target_idx - 1]
                        previous_price = hist.loc[actual_previous_date, 'Close']
                        logger.info(f"Found previous price on: {actual_previous_date}")
                
                return {
                    'yahoo_symbol': yahoo_symbol,
                    'target_date': target_date_str,
                    'actual_target_date': actual_target_date,
                    'actual_previous_date': actual_previous_date,
                    'target_price': float(target_price) if target_price is not None else None,
                    'previous_price': float(previous_price) if previous_price is not None else None,
                    'price_data': hist.to_dict('index'),
                    'data_available': target_price is not None and previous_price is not None,
                    'available_dates': hist_dates,
                    'date_range_requested': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                }
                
            except Exception as e:
                logger.warning(f"Error fetching prices for {yahoo_symbol}: {str(e)}")
                continue
        
        logger.error(f"Could not fetch price data for any symbol variation of {stock_code}")
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
        
        if not report_data['stock_code']:
            logger.warning(f"Missing stock code in {file_path}")
            return report_data
            
        if not report_data['publication_date']:
            logger.warning(f"Missing publication date in {file_path}")
            return report_data
        
        # Log the extracted information
        logger.info(f"Stock: {report_data['stock_code']}")
        logger.info(f"Publication date: {report_data['publication_date']}")
        logger.info(f"Timestamp field: {report_data['timestamp_field']}")
        logger.info(f"Dates found in content: {report_data['found_dates']}")
        
        # Determine target trading date
        target_date = self.get_target_trading_date(
            report_data['publication_date'], 
            report_data['stock_code']
        )
        
        logger.info(f"Target trading date: {target_date}")
        
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
        Process all JSON files in the notification directory
        
        Returns:
            List of processed reports
        """
        if not self.data_directory.exists():
            logger.error(f"Directory does not exist: {self.data_directory}")
            logger.info("Available notification folders:")
            available_folders = self.list_available_notification_folders()
            for folder in available_folders:
                logger.info(f"  - {folder}")
            return []
        
        json_files = list(self.data_directory.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files in {self.data_directory}")
        
        if len(json_files) == 0:
            logger.warning(f"No JSON files found in {self.data_directory}")
            return []
        
        results = []
        for file_path in json_files:
            result = self.process_single_report(str(file_path))
            if result:
                results.append(result)
        
        self.processed_data = results
        return results
    
    def save_processed_data(self, output_file: str = None):
        """
        Save processed data to JSON file
        
        Args:
            output_file: Output file name (default: processed_{stock_code}_data.json)
        """
        if not self.processed_data:
            logger.warning("No processed data to save")
            return
        
        # Default output filename
        if output_file is None:
            output_file = f"processed_{self.stock_code.lower()}_data.json"
        
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

    def test_stock_symbol_availability(self):
        """
        Test different stock symbol formats to see which one works
        ENHANCED: Better testing and error handling
        """
        test_symbols = self.get_alternative_yahoo_symbols(self.stock_code)
        
        print(f"\n=== Testing Stock Symbol Availability for {self.stock_code} ===")
        
        for symbol in test_symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # First check if symbol exists
                try:
                    info = ticker.info
                    if not info or len(info) < 5:  # Basic check if info is meaningful
                        print(f"❌ {symbol}: Symbol not found or invalid")
                        continue
                    else:
                        print(f"✅ {symbol}: Symbol exists")
                        if 'longName' in info:
                            print(f"   Company: {info['longName']}")
                        if 'currency' in info:
                            print(f"   Currency: {info['currency']}")
                except Exception as e:
                    print(f"❌ {symbol}: Error getting info - {str(e)}")
                    continue
                
                # Test with a broader date range in the past
                end_date = datetime.now() - timedelta(days=1)  # Yesterday
                start_date = end_date - timedelta(days=60)     # 60 days ago
                
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    print(f"✅ {symbol}: Historical data available ({len(hist)} records)")
                    print(f"   Date range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}")
                    print(f"   Latest price: {hist['Close'].iloc[-1]:.2f}")
                else:
                    print(f"❌ {symbol}: No historical data found")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {str(e)}")
        
        # Test with sample date from your JSON
        print(f"\n=== Testing with specific date (2025-06-04) ===")
        test_date = datetime(2025, 6, 4)
        for symbol in test_symbols:
            try:
                ticker = yf.Ticker(symbol)
                start_date = test_date - timedelta(days=10)
                end_date = test_date + timedelta(days=2)
                
                hist = ticker.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    print(f"✅ {symbol}: Found {len(hist)} records around {test_date.strftime('%Y-%m-%d')}")
                    print(f"   Available dates: {[d.strftime('%Y-%m-%d') for d in hist.index]}")
                else:
                    print(f"❌ {symbol}: No data around {test_date.strftime('%Y-%m-%d')}")
                    
            except Exception as e:
                print(f"❌ {symbol}: Error - {str(e)}")


# Example usage and testing
def main():
    """
    Main function to demonstrate the pipeline
    """
    print("=== KAP Report Processor ===")
    
    # Option 1: Auto-detect (if only one notification folder exists)
    try:
        processor = KAPReportProcessor()
        print(f"Auto-detected stock: {processor.stock_code}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Auto-detection failed: {e}")
        
        # Option 2: List available folders and let user choose
        temp_processor = KAPReportProcessor.__new__(KAPReportProcessor)
        temp_processor.project_directory = Path(".")
        available_folders = temp_processor.list_available_notification_folders()
        
        if available_folders:
            print("\nAvailable notification folders:")
            for folder in available_folders:
                stock_code = folder.replace("_notifications", "").upper()
                print(f"  - {folder} (Stock: {stock_code})")
            
            # Use the first available folder for demo
            first_stock = available_folders[0].replace("_notifications", "").upper()
            processor = KAPReportProcessor(stock_code=first_stock)
            print(f"\nUsing stock: {first_stock}")
        else:
            print("No notification folders found!")
            return None, None
    
    # Test stock symbol availability first
    processor.test_stock_symbol_availability()
    
    # Process all JSON files in the notification directory
    results = processor.process_directory()
    
    if not results:
        print("No data processed!")
        return processor, None
    
    # Save processed data
    processor.save_processed_data()
    
    # Get training dataset
    df = processor.get_training_dataset()
    
    if not df.empty:
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Total reports processed: {len(results)}")
        print(f"Reports with valid price data: {len(df)}")
        print(f"Stock processed: {processor.stock_code}")
        
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
        csv_filename = f"training_dataset_{processor.stock_code.lower()}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nTraining dataset saved to '{csv_filename}'")
    else:
        print("\n=== NO VALID PRICE DATA FOUND ===")
        print("This could be due to:")
        print("1. Stock symbol not available on Yahoo Finance")
        print("2. Date range issues")
        print("3. Market holidays or non-trading days")
        print("4. Delisted stock")
    
    return processor, df


if __name__ == "__main__":
    processor, df = main()