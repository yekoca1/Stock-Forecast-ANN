"""
Stage 4: Integration & Testing
Complete KAP Report Analysis Pipeline

This module integrates all stages and provides:
1. End-to-end pipeline testing
2. Prediction confidence scoring
3. Performance evaluation
4. Automated retraining capabilities
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Import your existing modules
try:
    from kap_processor import KAPReportProcessor
    # Note: These modules need to be created separately as they're not in the provided files
    # from nlp_processor import FinancialNLPProcessor  
    # from model_development import FinancialMLPipeline
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Make sure all your stage 1-3 modules are in the same directory")

class KAPPipeline:
    """
    Complete KAP Report Analysis Pipeline
    Integrates data processing, NLP analysis, and prediction models
    """
    
    def __init__(self, stock_code=None, project_directory=".", retrain_threshold=5):
        """
        Initialize the pipeline
        
        Args:
            stock_code: Stock code to process (e.g., 'THYAO')
            project_directory: Project root directory containing notification folders
            retrain_threshold: Minimum new reports needed to trigger retraining
        """
        self.stock_code = stock_code
        self.project_directory = Path(project_directory)
        self.retrain_threshold = retrain_threshold
        
        # Initialize KAP processor
        try:
            self.kap_processor = KAPReportProcessor(stock_code=stock_code, project_directory=project_directory)
            if not self.stock_code:
                self.stock_code = self.kap_processor.stock_code
        except Exception as e:
            print(f"Error initializing KAP processor: {e}")
            self.kap_processor = None
        
        # File paths (use stock code in filenames)
        stock_suffix = f"_{self.stock_code.lower()}" if self.stock_code else ""
        self.processed_data_file = f"processed{stock_suffix}_data.json"
        self.training_dataset_file = f"training_dataset{stock_suffix}.csv"
        self.model_file = f"trained_models{stock_suffix}.joblib"
        self.last_processed_file = f"last_processed{stock_suffix}.json"
        
        # Initialize other processors (placeholders - need to be implemented)
        self.nlp_processor = None
        self.model_dev = None
        
        # Models
        self.models = {}
        self.model_metrics = {}
        
        print("KAP Pipeline initialized")
        print(f"Stock code: {self.stock_code}")
        print(f"Data directory: {self.kap_processor.data_directory if self.kap_processor else 'Not found'}")
        print(f"Retrain threshold: {retrain_threshold} new reports")
    
    def _initialize_processors(self):
        """Initialize all processor modules"""
        try:
            if self.kap_processor is None:
                self.kap_processor = KAPReportProcessor(stock_code=self.stock_code, project_directory=str(self.project_directory))
            
            # TODO: Initialize NLP and ML processors when available
            # self.nlp_processor = FinancialNLPProcessor()
            # self.model_dev = FinancialMLPipeline()
            
            print("✅ KAP processor initialized successfully")
            if self.nlp_processor is None or self.model_dev is None:
                print("⚠️  NLP and ML processors not available - pipeline will be limited")
        except Exception as e:
            print(f"❌ Error initializing processors: {e}")
            raise
    
    def _check_for_new_data(self):
        """
        Check if there are new JSON files to process
        Returns: (has_new_data, new_files_count, total_files)
        """
        if not self.kap_processor or not self.kap_processor.data_directory.exists():
            print(f"❌ Data directory not found")
            return False, 0, 0
        
        # Get all JSON files
        json_files = list(self.kap_processor.data_directory.glob("*.json"))
        total_files = len(json_files)
        
        # Check last processed info
        last_processed_info = {}
        if os.path.exists(self.last_processed_file):
            try:
                with open(self.last_processed_file, 'r') as f:
                    last_processed_info = json.load(f)
            except:
                pass
        
        last_count = last_processed_info.get('processed_count', 0)
        new_files_count = total_files - last_count
        
        return new_files_count >= self.retrain_threshold, new_files_count, total_files
    
    def _update_processed_info(self, total_files):
        """Update the last processed file info"""
        info = {
            'processed_count': total_files,
            'last_processed_date': datetime.now().isoformat(),
            'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'stock_code': self.stock_code
        }
        
        with open(self.last_processed_file, 'w') as f:
            json.dump(info, f, indent=2)
    
    def process_all_data(self):
        """
        Process all JSON files through the complete pipeline
        Returns: processed_data list and training DataFrame
        """
        print("\n" + "="*50)
        print("PROCESSING ALL DATA")
        print("="*50)
        
        if self.kap_processor is None:
            self._initialize_processors()
        
        try:
            # Stage 1: Data Processing using KAPReportProcessor
            print("\n📊 Stage 1: Processing KAP reports...")
            processed_data = self.kap_processor.process_directory()
            
            if not processed_data or len(processed_data) == 0:
                print("❌ No data processed from Stage 1")
                return None, None
            
            print(f"✅ Processed {len(processed_data)} reports")
            
            # Save processed data
            self.kap_processor.save_processed_data(self.processed_data_file)
            
            # Get training dataset
            training_df = self.kap_processor.get_training_dataset()
            
            if training_df.empty:
                print("❌ No valid training data created")
                return processed_data, None
            
            # Save training dataset
            training_df.to_csv(self.training_dataset_file, index=False)
            print(f"✅ Training dataset saved to {self.training_dataset_file}")
            
            # Stage 2: NLP Processing (if available)
            if self.nlp_processor:
                print("\n🔤 Stage 2: NLP analysis...")
                # TODO: Implement NLP processing
                # nlp_features = self.nlp_processor.process_batch(training_df)
                # training_df = pd.concat([training_df, nlp_features], axis=1)
                pass
            else:
                print("\n⚠️  Stage 2: NLP processor not available - skipping")
            
            return processed_data, training_df
            
        except Exception as e:
            print(f"❌ Error in data processing: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_models(self, training_df=None):
        """
        Train prediction models
        
        Args:
            training_df: DataFrame with processed data (if None, loads from file)
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        if training_df is None:
            if not os.path.exists(self.training_dataset_file):
                print("❌ No training dataset found. Run process_all_data() first.")
                return False
            
            training_df = pd.read_csv(self.training_dataset_file)
        
        if training_df.empty:
            print("❌ Training dataset is empty")
            return False
        
        try:
            # Basic model training (placeholder until ML module is available)
            print(f"\n🤖 Training models on {len(training_df)} samples...")
            
            if self.model_dev:
                # TODO: Use actual ML pipeline when available
                # models, metrics = self.model_dev.train_all_models(training_df)
                # self.models = models
                # self.model_metrics = metrics
                pass
            else:
                # Create dummy models for testing
                print("⚠️  ML pipeline not available - creating placeholder models")
                
                # Simple baseline models
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, accuracy_score
                
                if 'price_change_percentage' in training_df.columns:
                    # Prepare features (numeric columns only)
                    feature_cols = ['content_length', 'target_price', 'previous_price']
                    available_features = [col for col in feature_cols if col in training_df.columns]
                    
                    if available_features:
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
                        reg_model = RandomForestRegressor(n_estimators=50, random_state=42)
                        reg_model.fit(X_train, y_reg_train)
                        reg_pred = reg_model.predict(X_test)
                        
                        # Train classification model
                        clf_model = RandomForestClassifier(n_estimators=50, random_state=42)
                        clf_model.fit(X_train, y_clf_train)
                        clf_pred = clf_model.predict(X_test)
                        
                        self.models = {
                            'regression': reg_model,
                            'classification': clf_model
                        }
                        
                        self.model_metrics = {
                            'regression': {
                                'mae': mean_absolute_error(y_reg_test, reg_pred),
                                'feature_cols': available_features
                            },
                            'classification': {
                                'accuracy': accuracy_score(y_clf_test, clf_pred),
                                'feature_cols': available_features
                            }
                        }
            
            # Save models
            model_data = {
                'models': self.models,
                'metrics': self.model_metrics,
                'training_date': datetime.now().isoformat(),
                'data_shape': training_df.shape,
                'stock_code': self.stock_code
            }
            
            joblib.dump(model_data, self.model_file)
            print(f"✅ Models saved to {self.model_file}")
            
            # Print performance summary
            self._print_model_performance()
            
            return True
            
        except Exception as e:
            print(f"❌ Error in model training: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_models(self):
        """Load trained models from file"""
        if not os.path.exists(self.model_file):
            print("❌ No trained models found. Run train_models() first.")
            return False
        
        try:
            model_data = joblib.load(self.model_file)
            self.models = model_data['models']
            self.model_metrics = model_data['metrics']
            
            print(f"✅ Models loaded from {self.model_file}")
            print(f"   Training date: {model_data.get('training_date', 'Unknown')}")
            print(f"   Training data shape: {model_data.get('data_shape', 'Unknown')}")
            print(f"   Stock code: {model_data.get('stock_code', 'Unknown')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_single_report(self, json_file_path, confidence_level=0.95):
        """
        Predict impact of a single KAP report
        
        Args:
            json_file_path: Path to the JSON report file
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            dict: Prediction results with confidence scores
        """
        if not self.models:
            if not self.load_models():
                return None
        
        if self.kap_processor is None:
            self._initialize_processors()
        
        try:
            print(f"\n🔍 Analyzing report: {os.path.basename(json_file_path)}")
            
            # Process single report
            report_data = self.kap_processor.process_single_report(json_file_path)
            if report_data is None:
                print("❌ Failed to process report")
                return None
            
            print(f"✅ Report processed successfully")
            print(f"   Stock: {report_data.get('stock_code', 'Unknown')}")
            print(f"   Publication date: {report_data.get('publication_date', 'Unknown')}")
            
            # Check if we have valid price data
            if report_data.get('price_change_percentage') is None:
                print("⚠️  No price data available - making prediction based on content only")
            
            # Prepare features for prediction
            feature_cols = self.model_metrics.get('regression', {}).get('feature_cols', [])
            if not feature_cols:
                print("❌ No feature columns found in trained model")
                return None
            
            # Create feature vector
            features = {}
            for col in feature_cols:
                if col == 'content_length':
                    features[col] = len(report_data.get('content', ''))
                elif col in report_data:
                    features[col] = report_data[col]
                elif report_data.get('price_data') and col in report_data['price_data']:
                    features[col] = report_data['price_data'][col]
                else:
                    features[col] = 0  # Default value
            
            feature_df = pd.DataFrame([features])
            
            # Make predictions
            results = {}
            
            # Regression prediction (price change)
            if 'regression' in self.models:
                reg_model = self.models['regression']
                price_pred = reg_model.predict(feature_df)[0]
                
                # Calculate confidence interval (simple approach)
                reg_mae = self.model_metrics.get('regression', {}).get('mae', 1.0)
                z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
                
                confidence_interval = (
                    price_pred - z_score * reg_mae,
                    price_pred + z_score * reg_mae
                )
                
                results['price_change_prediction'] = {
                    'predicted_change_%': round(price_pred, 3),
                    'confidence_interval_%': [round(ci, 3) for ci in confidence_interval],
                    'confidence_level': confidence_level
                }
            
            # Classification prediction (direction)
            if 'classification' in self.models:
                clf_model = self.models['classification']
                direction_pred = clf_model.predict(feature_df)[0]
                direction_proba = clf_model.predict_proba(feature_df)[0]
                
                results['direction_prediction'] = {
                    'predicted_direction': 'UP' if direction_pred == 1 else 'DOWN',
                    'confidence_score': round(max(direction_proba), 3),
                    'probabilities': {
                        'DOWN': round(direction_proba[0], 3),
                        'UP': round(direction_proba[1], 3) if len(direction_proba) > 1 else 0.0
                    }
                }
            
            # Add report metadata
            results['report_info'] = {
                'stock_code': report_data.get('stock_code', 'Unknown'),
                'publication_date': str(report_data.get('publication_date', 'Unknown')),
                'analysis_timestamp': datetime.now().isoformat(),
                'features_used': list(features.keys())
            }
            
            # Add actual price change if available
            if report_data.get('price_change_percentage') is not None:
                results['actual_data'] = {
                    'actual_change_%': report_data['price_change_percentage'],
                    'actual_direction': 'UP' if report_data.get('direction', 0) > 0 else 'DOWN',
                    'target_price': report_data.get('price_data', {}).get('target_price'),
                    'previous_price': report_data.get('price_data', {}).get('previous_price')
                }
            
            self._print_prediction_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def batch_predict(self, json_files_dir=None):
        """
        Predict impacts for all reports in a directory
        
        Args:
            json_files_dir: Directory containing JSON files (default: data directory)
        """
        if json_files_dir is None:
            json_files_dir = self.kap_processor.data_directory
        
        if not os.path.exists(json_files_dir):
            print(f"❌ Directory {json_files_dir} not found")
            return []
        
        json_files = list(Path(json_files_dir).glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {json_files_dir}")
            return []
        
        print(f"\n📋 Batch prediction for {len(json_files)} reports")
        print("="*50)
        
        results = []
        successful = 0
        
        for json_file in json_files:
            result = self.predict_single_report(str(json_file))
            
            if result:
                results.append(result)
                successful += 1
            
            print("-" * 30)
        
        print(f"\n✅ Batch prediction completed")
        print(f"   Successful predictions: {successful}/{len(json_files)}")
        
        return results
    
    def run_full_pipeline(self, force_retrain=False):
        """
        Run the complete pipeline with automatic retraining logic
        
        Args:
            force_retrain: Force retraining even if threshold not met
        """
        print("\n" + "="*60)
        print("KAP REPORT ANALYSIS - FULL PIPELINE")
        print("="*60)
        
        # Check for new data
        has_new_data, new_files, total_files = self._check_for_new_data()
        
        print(f"📁 Data status:")
        print(f"   Total JSON files: {total_files}")
        print(f"   New files since last run: {new_files}")
        print(f"   Retrain threshold: {self.retrain_threshold}")
        
        should_retrain = force_retrain or has_new_data
        
        if should_retrain:
            print(f"\n🔄 Retraining triggered ({'forced' if force_retrain else 'automatic'})")
            
            # Process all data
            processed_data, training_df = self.process_all_data()
            if processed_data is None:
                print("❌ Pipeline failed at data processing stage")
                return False
            
            # Train models
            if not self.train_models(training_df):
                print("❌ Pipeline failed at model training stage")
                return False
            
            # Update processed info
            self._update_processed_info(total_files)
        
        else:
            print(f"\n📚 Using existing models (no retraining needed)")
            if not self.load_models():
                print("❌ No existing models found. Running full pipeline...")
                return self.run_full_pipeline(force_retrain=True)
        
        # Run batch predictions
        print(f"\n🎯 Running predictions on all reports...")
        batch_results = self.batch_predict()
        
        # Performance summary
        self._print_pipeline_summary(batch_results)
        
        print(f"\n✅ Full pipeline completed successfully!")
        return True
    
    def test_stock_symbol(self):
        """Test stock symbol availability using KAP processor"""
        if self.kap_processor:
            self.kap_processor.test_stock_symbol_availability()
        else:
            print("❌ KAP processor not available")
    
    def _print_model_performance(self):
        """Print model performance metrics"""
        print(f"\n📈 Model Performance Summary:")
        print("-" * 40)
        
        for model_type, metrics in self.model_metrics.items():
            print(f"\n{model_type.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric.upper()}: {value:.4f}")
                else:
                    print(f"  {metric.upper()}: {value}")
    
    def _print_prediction_results(self, results):
        """Print prediction results in a formatted way"""
        print(f"\n🎯 Prediction Results:")
        print("-" * 30)
        
        info = results.get('report_info', {})
        print(f"Stock: {info.get('stock_code', 'Unknown')}")
        print(f"Date: {info.get('publication_date', 'Unknown')}")
        
        if 'price_change_prediction' in results:
            pred = results['price_change_prediction']
            print(f"\n💰 Price Change Prediction:")
            print(f"  Predicted: {pred['predicted_change_%']}%")
            print(f"  Confidence Interval: {pred['confidence_interval_%']} (95%)")
        
        if 'direction_prediction' in results:
            pred = results['direction_prediction']
            print(f"\n📈 Direction Prediction:")
            print(f"  Direction: {pred['predicted_direction']}")
            print(f"  Confidence: {pred['confidence_score']} ({pred['confidence_score']*100:.1f}%)")
        
        if 'actual_data' in results:
            actual = results['actual_data']
            print(f"\n📊 Actual Data (for comparison):")
            print(f"  Actual change: {actual['actual_change_%']}%")
            print(f"  Actual direction: {actual['actual_direction']}")
    
    def _print_pipeline_summary(self, batch_results):
        """Print pipeline execution summary"""
        if not batch_results:
            return
        
        print(f"\n📊 Pipeline Execution Summary:")
        print("-" * 40)
        print(f"Total predictions: {len(batch_results)}")
        print(f"Stock: {self.stock_code}")
        
        # Direction predictions summary
        directions = []
        confidences = []
        actual_directions = []
        
        for result in batch_results:
            if 'direction_prediction' in result:
                directions.append(result['direction_prediction']['predicted_direction'])
                confidences.append(result['direction_prediction']['confidence_score'])
            
            if 'actual_data' in result:
                actual_directions.append(result['actual_data']['actual_direction'])
        
        if directions:
            up_count = directions.count('UP')
            down_count = directions.count('DOWN')
            avg_confidence = np.mean(confidences)
            
            print(f"\nDirection predictions:")
            print(f"  UP: {up_count} ({up_count/len(directions)*100:.1f}%)")
            print(f"  DOWN: {down_count} ({down_count/len(directions)*100:.1f}%)")
            print(f"  Average confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
        
        # Accuracy calculation if actual data is available
        if actual_directions and len(actual_directions) == len(directions):
            correct = sum(1 for pred, actual in zip(directions, actual_directions) if pred == actual)
            accuracy = correct / len(directions)
            print(f"\nPrediction Accuracy:")
            print(f"  Correct predictions: {correct}/{len(directions)} ({accuracy*100:.1f}%)")


# Example usage and testing functions
def test_pipeline(stock_code=None):
    """Test the complete pipeline"""
    print("Testing KAP Pipeline...")
    
    try:
        pipeline = KAPPipeline(stock_code=stock_code)
        
        # Test stock symbol first
        print("\n🧪 Testing stock symbol availability...")
        pipeline.test_stock_symbol()
        
        # Test full pipeline
        success = pipeline.run_full_pipeline(force_retrain=True)
        
        if success:
            print("\n✅ Pipeline test completed successfully!")
        else:
            print("\n❌ Pipeline test failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_new_report(json_file_path, stock_code=None):
    """Quick function to predict impact of a new report"""
    try:
        pipeline = KAPPipeline(stock_code=stock_code)
        
        # Load existing models
        if not pipeline.load_models():
            print("No trained models found. Training first...")
            pipeline.run_full_pipeline(force_retrain=True)
        
        # Make prediction
        result = pipeline.predict_single_report(json_file_path)
        return result
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return None

def list_available_stocks():
    """List all available stock notification folders"""
    try:
        temp_processor = KAPReportProcessor()
        folders = temp_processor.list_available_notification_folders()
        
        print("Available stock notification folders:")
        for folder in folders:
            stock_code = folder.replace("_notifications", "").upper()
            print(f"  - {folder} (Stock: {stock_code})")
        
        return folders
        
    except Exception as e:
        print(f"Error listing folders: {e}")
        return []

if __name__ == "__main__":
    # Show available stocks
    print("="*60)
    print("KAP PIPELINE - AVAILABLE STOCKS")
    print("="*60)
    available_stocks = list_available_stocks()
    
    if available_stocks:
        # Test with first available stock
        first_stock = available_stocks[0].replace("_notifications", "").upper()
        print(f"\nTesting with stock: {first_stock}")
        test_success = test_pipeline(stock_code=first_stock)
    else:
        print("No stocks available for testing")
        test_success = False
    
    print("\n" + "="*60)
    if test_success:
        print("PIPELINE READY FOR USE")
    else:
        print("PIPELINE SETUP INCOMPLETE")
    print("="*60)
    print("Usage examples:")
    print("1. List stocks: list_available_stocks()")
    print("2. Full pipeline: pipeline = KAPPipeline('THYAO'); pipeline.run_full_pipeline()")
    print("3. Single prediction: predict_new_report('path/to/report.json', 'THYAO')")
    print("4. Test stock symbol: pipeline.test_stock_symbol()")