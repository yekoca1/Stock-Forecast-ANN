"""
Optimized KAP Report Analysis Pipeline
Predicts tomorrow's closing price and maintains historical predictions
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, r2_score
import yfinance as yf

warnings.filterwarnings('ignore')

from kap_processor import KAPReportProcessor

class OptimizedKAPPipeline:
    """Streamlined KAP Pipeline for Historical Training and Tomorrow's Price Prediction"""
    
    def __init__(self, stock_code=None, project_directory=".", retrain_threshold=5):
        self.stock_code = stock_code
        self.project_directory = Path(project_directory)
        self.retrain_threshold = retrain_threshold
        
        # Initialize KAP processor
        self.kap_processor = self._init_kap_processor()
        
        # File paths
        self._setup_file_paths()
        
        # Model storage
        self.models = {}
        self.model_metrics = {}
        self.historical_predictions = []
        
        print(f"Pipeline initialized for {self.stock_code}")
    
    def _init_kap_processor(self):
        """Initialize KAP processor with error handling"""
        try:
            processor = KAPReportProcessor(stock_code=self.stock_code, project_directory=self.project_directory)
            if not self.stock_code:
                self.stock_code = processor.stock_code
            return processor
        except Exception as e:
            print(f"KAP processor initialization failed: {e}")
            return None
    
    def _setup_file_paths(self):
        """Setup file paths with stock-specific naming"""
        suffix = f"_{self.stock_code.lower()}" if self.stock_code else ""
        self.training_file = f"training_dataset{suffix}.csv"
        self.model_file = f"trained_models{suffix}.joblib"
        self.predictions_file = f"predictions{suffix}.json"
        self.last_processed_file = f"last_processed{suffix}.json"
    
    def _get_current_price(self):
        """Get current stock price using yfinance"""
        try:
            ticker = yf.Ticker(f"{self.stock_code}.IS")  # Turkish stock market
            hist = ticker.history(period="5d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except:
            pass
        return None
    
    def _check_new_data(self):
        """Check for new data and determine if retraining is needed"""
        if not self.kap_processor or not self.kap_processor.data_directory.exists():
            return False, 0
        
        current_files = len(list(self.kap_processor.data_directory.glob("*.json")))
        
        try:
            with open(self.last_processed_file, 'r') as f:
                last_info = json.load(f)
            last_count = last_info.get('processed_count', 0)
        except:
            last_count = 0
        
        new_files = current_files - last_count
        return new_files >= self.retrain_threshold, new_files
    
    def _update_processed_info(self, file_count):
        """Update processing metadata"""
        info = {
            'processed_count': file_count,
            'last_processed': datetime.now().isoformat(),
            'stock_code': self.stock_code
        }
        with open(self.last_processed_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def process_data(self):
        """Process all JSON files and create training dataset"""
        if not self.kap_processor:
            return None
        
        try:
            # Process all reports
            processed_data = self.kap_processor.process_directory()
            if not processed_data:
                return None
            
            # Get training dataset
            training_df = self.kap_processor.get_training_dataset()
            if training_df.empty:
                return None
            
            # Save training data
            training_df.to_csv(self.training_file, index=False)
            print(f"Training dataset saved: {len(training_df)} records")
            
            return training_df
            
        except Exception as e:
            print(f"Data processing error: {e}")
            return None
    
    def train_models(self, training_df):
        """Train regression and classification models"""
        try:
            # Prepare features
            feature_cols = ['content_length', 'target_price', 'previous_price']
            available_features = [col for col in feature_cols if col in training_df.columns]
            
            if not available_features:
                print("No valid features found")
                return False
            
            # Prepare data
            X = training_df[available_features].fillna(0)
            y_reg = training_df['price_change_percentage']
            y_clf = training_df['direction']
            
            # Split data
            X_train, X_test, y_reg_train, y_reg_test = train_test_split(
                X, y_reg, test_size=0.2, random_state=42
            )
            _, _, y_clf_train, y_clf_test = train_test_split(
                X, y_clf, test_size=0.2, random_state=42
            )
            
            # Train regression model
            reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
            reg_model.fit(X_train, y_reg_train)
            reg_pred = reg_model.predict(X_test)
            
            # Train classification model
            clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_model.fit(X_train, y_clf_train)
            clf_pred = clf_model.predict(X_test)
            
            # Store models and metrics
            self.models = {
                'regression': reg_model,
                'classification': clf_model,
                'feature_cols': available_features
            }
            
            self.model_metrics = {
                'regression': {
                    'mae': mean_absolute_error(y_reg_test, reg_pred),
                    'r2': r2_score(y_reg_test, reg_pred),
                    'mse': np.mean((y_reg_test - reg_pred) ** 2)
                },
                'classification': {
                    'accuracy': accuracy_score(y_clf_test, clf_pred)
                }
            }
            
            # Generate historical predictions for training data
            self._generate_historical_predictions(X, y_reg, y_clf)
            
            # Save models
            self._save_models()
            
            print(f"Models trained successfully")
            print(f"Regression R²: {self.model_metrics['regression']['r2']:.3f}")
            print(f"Classification Accuracy: {self.model_metrics['classification']['accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Model training error: {e}")
            return False
    
    def _generate_historical_predictions(self, X, y_reg_actual, y_clf_actual):
        """Generate predictions for historical data (for training evaluation)"""
        try:
            reg_predictions = self.models['regression'].predict(X)
            clf_predictions = self.models['classification'].predict(X)
            clf_probabilities = self.models['classification'].predict_proba(X)
            
            self.historical_predictions = []
            
            for i in range(len(X)):
                pred = {
                    'index': i,
                    'predicted_change_%': round(float(reg_predictions[i]), 3),
                    'actual_change_%': round(float(y_reg_actual.iloc[i]), 3),
                    'predicted_direction': 'UP' if clf_predictions[i] == 1 else 'DOWN',
                    'actual_direction': 'UP' if y_clf_actual.iloc[i] == 1 else 'DOWN',
                    'direction_confidence': round(float(max(clf_probabilities[i])), 3),
                    'prediction_error': round(float(abs(reg_predictions[i] - y_reg_actual.iloc[i])), 3)
                }
                self.historical_predictions.append(pred)
            
            print(f"Generated {len(self.historical_predictions)} historical predictions")
            
        except Exception as e:
            print(f"Historical prediction generation error: {e}")
    
    def _save_models(self):
        """Save trained models and metadata"""
        model_data = {
            'models': self.models,
            'metrics': self.model_metrics,
            'historical_predictions': self.historical_predictions,
            'training_date': datetime.now().isoformat(),
            'stock_code': self.stock_code
        }
        joblib.dump(model_data, self.model_file)
    
    def load_models(self):
        """Load trained models from file"""
        if not os.path.exists(self.model_file):
            return False
        
        try:
            model_data = joblib.load(self.model_file)
            self.models = model_data['models']
            self.model_metrics = model_data['metrics']
            self.historical_predictions = model_data.get('historical_predictions', [])
            
            # Handle legacy model format - extract feature_cols from metrics if not in models
            if 'feature_cols' not in self.models and 'regression' in self.model_metrics:
                legacy_features = self.model_metrics['regression'].get('feature_cols', 
                                                                     ['content_length', 'target_price', 'previous_price'])
                self.models['feature_cols'] = legacy_features
            
            return True
        except Exception as e:
            print(f"Model loading error: {e}")
            return False
    
    def predict_tomorrow_price(self):
        """Predict tomorrow's closing price"""
        if not self.models:
            print("No models available for prediction")
            return None
        
        try:
            # Get current price
            current_price = self._get_current_price()
            if not current_price:
                print("Warning: Could not fetch current price, using estimated value")
                current_price = 100.0  # Fallback value
            
            # Get feature columns - handle both new and legacy formats
            feature_cols = self.models.get('feature_cols', ['content_length', 'target_price', 'previous_price'])
            
            # Create feature vector for tomorrow's prediction
            features = {}
            for col in feature_cols:
                if col == 'content_length':
                    features[col] = 1000  # Average content length
                elif col == 'target_price':
                    features[col] = current_price
                elif col == 'previous_price':
                    features[col] = current_price
                else:
                    features[col] = 0
            
            feature_df = pd.DataFrame([features])
            
            # Make predictions
            reg_model = self.models.get('regression')
            clf_model = self.models.get('classification')
            
            if not reg_model or not clf_model:
                print("Required models not found")
                return None
            
            predicted_change = reg_model.predict(feature_df)[0]
            predicted_direction = clf_model.predict(feature_df)[0]
            direction_proba = clf_model.predict_proba(feature_df)[0]
            
            # Calculate tomorrow's price
            tomorrow_price = current_price * (1 + predicted_change / 100)
            
            # Calculate confidence interval
            mae = self.model_metrics.get('regression', {}).get('mae', 5.0)
            confidence_interval = [
                current_price * (1 + (predicted_change - 1.96 * mae) / 100),
                current_price * (1 + (predicted_change + 1.96 * mae) / 100)
            ]
            
            tomorrow_prediction = {
                'prediction_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'current_price': round(current_price, 2),
                'predicted_price': round(tomorrow_price, 2),
                'predicted_change_%': round(predicted_change, 3),
                'predicted_direction': 'UP' if predicted_direction == 1 else 'DOWN',
                'direction_confidence': round(max(direction_proba), 3),
                'confidence_interval': [round(ci, 2) for ci in confidence_interval],
                'model_accuracy': {
                    'regression_r2': round(self.model_metrics.get('regression', {}).get('r2', 0), 3),
                    'classification_accuracy': round(self.model_metrics.get('classification', {}).get('accuracy', 0), 3)
                },
                'features_used': feature_cols
            }
            
            return tomorrow_prediction
            
        except Exception as e:
            print(f"Tomorrow prediction error: {e}")
            print(f"Available model keys: {list(self.models.keys()) if self.models else 'None'}")
            return None
    
    def generate_full_report(self):
        """Generate comprehensive prediction report"""
        # Always generate report with available data
        report = {
            'stock_code': self.stock_code,
            'analysis_timestamp': datetime.now().isoformat(),
            'tomorrow_prediction': None,
            'historical_predictions': {
                'total_predictions': len(self.historical_predictions),
                'sample_predictions': self.historical_predictions[:10] if self.historical_predictions else [],
                'summary_stats': self._calculate_historical_stats()
            },
            'model_performance': self.model_metrics,
            'model_info': {
                'models_available': list(self.models.keys()) if self.models else [],
                'training_features': self.models.get('feature_cols', []) if self.models else []
            }
        }
        
        # Try to get tomorrow's prediction
        tomorrow_pred = self.predict_tomorrow_price()
        if tomorrow_pred:
            report['tomorrow_prediction'] = tomorrow_pred
            print("✓ Tomorrow's prediction generated successfully")
        else:
            print("✗ Tomorrow's prediction failed, but showing available data")
            report['tomorrow_prediction_error'] = "Could not generate tomorrow's prediction"
        
        # Save report
        with open(self.predictions_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_historical_stats(self):
        """Calculate summary statistics for historical predictions"""
        if not self.historical_predictions:
            return {}
        
        errors = [pred['prediction_error'] for pred in self.historical_predictions]
        direction_correct = sum(1 for pred in self.historical_predictions 
                              if pred['predicted_direction'] == pred['actual_direction'])
        
        return {
            'mean_absolute_error': round(np.mean(errors), 3),
            'max_error': round(max(errors), 3),
            'min_error': round(min(errors), 3),
            'direction_accuracy': round(direction_correct / len(self.historical_predictions), 3)
        }
    
    def run_pipeline(self, force_retrain=False):
        """Execute the complete pipeline"""
        print(f"Starting KAP Pipeline for {self.stock_code}")
        
        # Check for new data
        needs_retrain, new_files = self._check_new_data()
        should_retrain = force_retrain or needs_retrain
        
        if should_retrain:
            print(f"Retraining with {new_files} new files...")
            
            # Process data
            training_df = self.process_data()
            if training_df is None:
                print("Data processing failed")
                return None
            
            # Train models
            if not self.train_models(training_df):
                print("Model training failed")
                return None
            
            # Update processed info
            file_count = len(list(self.kap_processor.data_directory.glob("*.json")))
            self._update_processed_info(file_count)
        
        else:
            print("Loading existing models...")
            if not self.load_models():
                print("No existing models found. Training new models...")
                return self.run_pipeline(force_retrain=True)
        
        # Always generate report (even if tomorrow prediction fails)
        report = self.generate_full_report()
        
        if report:
            print("\n=== PIPELINE RESULTS ===")
            
            # Show historical predictions summary
            hist_stats = report['historical_predictions']['summary_stats']
            if hist_stats:
                print(f"Historical Predictions: {report['historical_predictions']['total_predictions']}")
                print(f"Average Prediction Error: {hist_stats.get('mean_absolute_error', 'N/A')}")
                print(f"Direction Accuracy: {hist_stats.get('direction_accuracy', 'N/A')}")
            
            # Show model performance
            if report['model_performance']:
                reg_metrics = report['model_performance'].get('regression', {})
                clf_metrics = report['model_performance'].get('classification', {})
                print(f"Model R² Score: {reg_metrics.get('r2', 'N/A')}")
                print(f"Model Classification Accuracy: {clf_metrics.get('accuracy', 'N/A')}")
            
            # Show tomorrow's prediction if available
            if report['tomorrow_prediction']:
                tomorrow = report['tomorrow_prediction']
                print(f"\n=== TOMORROW'S PREDICTION ===")
                print(f"Current Price: {tomorrow['current_price']}")
                print(f"Predicted Price: {tomorrow['predicted_price']}")
                print(f"Expected Change: {tomorrow['predicted_change_%']}%")
                print(f"Direction: {tomorrow['predicted_direction']}")
                print(f"Confidence: {tomorrow['direction_confidence']}")
            else:
                print(f"\n=== TOMORROW'S PREDICTION ===")
                print("Prediction failed - see error details above")
            
            print(f"\nFull report saved to: {self.predictions_file}")
        
        return report


# Utility functions
def quick_prediction(stock_code):
    """Quick prediction for a specific stock"""
    try:
        pipeline = OptimizedKAPPipeline(stock_code=stock_code)
        return pipeline.run_pipeline()
    except Exception as e:
        print(f"Quick prediction error: {e}")
        return None

def list_available_stocks():
    """List available stock folders"""
    try:
        temp_processor = KAPReportProcessor()
        folders = temp_processor.list_available_notification_folders()
        return [folder.replace("_notifications", "").upper() for folder in folders]
    except:
        return []

if __name__ == "__main__":
    # Demo usage
    available_stocks = list_available_stocks()
    
    if available_stocks:
        print(f"Available stocks: {available_stocks}")
        
        # Test with first available stock
        test_stock = available_stocks[0]
        print(f"\nTesting with {test_stock}...")
        
        result = quick_prediction(test_stock)
        
        if result:
            print("\n=== PREDICTION SUMMARY ===")
            tomorrow = result['tomorrow_prediction']
            print(f"Current Price: {tomorrow['current_price']}")
            print(f"Predicted Price: {tomorrow['predicted_price']}")
            print(f"Expected Change: {tomorrow['predicted_change_%']}%")
            print(f"Direction: {tomorrow['predicted_direction']}")
            print(f"Confidence: {tomorrow['direction_confidence']}")
        else:
            print("Prediction failed")
    else:
        print("No stock data available")