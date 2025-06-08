"""
KAP Pipeline Performance Evaluator - Cleaned Version
Comprehensive testing and evaluation utilities for the KAP Pipeline
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_absolute_error, mean_squared_error, r2_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

class PipelineEvaluator:
    """Comprehensive evaluation system for the KAP Pipeline"""
    
    def __init__(self, pipeline=None):
        self.pipeline = pipeline
        self.results = {}
        
    def evaluate_model_performance(self, test_data=None, save_plots=True):
        """Evaluate model performance on test data"""
        print("\n" + "="*50)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*50)
        
        if not self._validate_pipeline():
            return None
            
        test_data = self._prepare_test_data(test_data)
        if test_data is None:
            return None
            
        results = {}
        X_test = self._prepare_features(test_data)
        
        # Evaluate models
        if self._has_model('regression') and 'price_change_percent' in test_data.columns:
            results['regression'] = self._evaluate_regression(X_test, test_data['price_change_percent'])
            if save_plots:
                self._plot_regression_results(results['regression'])
        
        if self._has_model('classification') and 'price_direction' in test_data.columns:
            results['classification'] = self._evaluate_classification(X_test, test_data['price_direction'])
            if save_plots:
                self._plot_classification_results(results['classification'])
        
        results['summary'] = self._generate_summary(results)
        self.results = results
        
        self._save_results(results, 'evaluation_results.json')
        self._print_evaluation_summary(results)
        
        return results
    
    def test_pipeline_reliability(self, num_tests=5, data_dir=None):
        """Test pipeline reliability with multiple runs"""
        print("\n" + "="*50)
        print("PIPELINE RELIABILITY TESTING")
        print("="*50)
        
        if not self.pipeline:
            print("❌ No pipeline provided for testing")
            return None
        
        test_files = self._find_test_files(data_dir)
        if not test_files:
            return None
            
        results = {'processing_times': [], 'error_count': 0, 'success_count': 0}
        
        print(f"🧪 Running {num_tests} reliability tests...")
        
        for i in range(num_tests):
            try:
                print(f"Test {i+1}/{num_tests}...", end=' ')
                test_file = np.random.choice(test_files)
                
                start_time = datetime.now()
                success = self._test_single_file(test_file)
                end_time = datetime.now()
                
                processing_time = (end_time - start_time).total_seconds()
                results['processing_times'].append(processing_time)
                
                if success:
                    results['success_count'] += 1
                    print("✅")
                else:
                    results['error_count'] += 1
                    print("❌")
                    
            except Exception as e:
                results['error_count'] += 1
                print(f"❌ Error: {str(e)[:50]}...")
        
        results['success_rate'] = results['success_count'] / num_tests
        results['avg_processing_time'] = np.mean(results['processing_times']) if results['processing_times'] else 0
        
        self._print_reliability_summary(results)
        return results
    
    def generate_monitoring_report(self):
        """Generate comprehensive performance monitoring report"""
        print("\n" + "="*50)
        print("PERFORMANCE MONITORING REPORT")
        print("="*50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_quality': self._check_data_quality(),
            'model_health': self._check_model_health(),
        }
        
        report['system_status'] = self._assess_system_status(report)
        report['recommendations'] = self._generate_recommendations(report)
        
        self._save_monitoring_report(report)
        self._print_monitoring_report(report)
        
        return report
    
    # Private helper methods
    def _validate_pipeline(self):
        """Validate pipeline and models exist"""
        if not self.pipeline or not hasattr(self.pipeline, 'models') or not self.pipeline.models:
            print("❌ No trained models found for evaluation")
            return False
        return True
    
    def _prepare_test_data(self, test_data):
        """Prepare test data for evaluation"""
        if test_data is not None:
            return test_data
            
        if not os.path.exists("processed_training_data.csv"):
            print("❌ No processed data found")
            return None
        
        full_data = pd.read_csv("processed_training_data.csv")
        split_idx = int(len(full_data) * 0.8)
        test_data = full_data[split_idx:].copy()
        
        print(f"📊 Using last {len(test_data)} records as test data")
        return test_data
    
    def _prepare_features(self, data):
        """Prepare feature matrix from data"""
        numeric_features = data.select_dtypes(include=[np.number]).columns
        return data[numeric_features]
    
    def _has_model(self, model_type):
        """Check if specific model type exists"""
        return model_type in self.pipeline.models
    
    def _evaluate_regression(self, X_test, y_test):
        """Evaluate regression model"""
        model = self.pipeline.models['regression']
        predictions = model.predict(X_test)
        
        # Calculate directional accuracy
        actual_direction = np.sign(y_test)
        pred_direction = np.sign(predictions)
        
        return {
            'predictions': predictions,
            'actual': y_test.values,
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions),
            'directional_accuracy': accuracy_score(actual_direction, pred_direction),
            'mape': np.mean(np.abs((y_test - predictions) / np.where(y_test != 0, y_test, 1))) * 100
        }
    
    def _evaluate_classification(self, X_test, y_test):
        """Evaluate classification model"""
        model = self.pipeline.models['classification']
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'actual': y_test.values,
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
            'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
            'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
    
    def _find_test_files(self, data_dir):
        """Find test files for reliability testing"""
        possible_dirs = [data_dir] if data_dir else ['data', 'json_data', 'reports', '.']
        
        for dir_name in possible_dirs:
            if dir_name and os.path.exists(dir_name):
                json_files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith('.json')]
                if json_files:
                    print(f"📁 Found {len(json_files)} test files in: {dir_name}")
                    return json_files
        
        print("❌ No JSON test files found")
        return None
    
    def _test_single_file(self, file_path):
        """Test processing of a single file"""
        try:
            if hasattr(self.pipeline, 'predict_single_report'):
                result = self.pipeline.predict_single_report(file_path)
            elif hasattr(self.pipeline, 'predict'):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                result = self.pipeline.predict(data)
            else:
                return False
            
            return result is not None
        except Exception:
            return False
    
    def _check_data_quality(self):
        """Check quality of processed data"""
        quality = {'data_exists': False, 'record_count': 0, 'issues': []}
        
        if os.path.exists("processed_training_data.csv"):
            quality['data_exists'] = True
            try:
                data = pd.read_csv("processed_training_data.csv")
                quality['record_count'] = len(data)
                quality['missing_values'] = data.isnull().sum().sum()
                
                if quality['record_count'] < 10:
                    quality['issues'].append("Very small dataset (< 10 records)")
                if quality['missing_values'] > quality['record_count'] * 0.1:
                    quality['issues'].append("High missing value rate (> 10%)")
                    
            except Exception as e:
                quality['issues'].append(f"Error reading data: {str(e)}")
        else:
            quality['issues'].append("No processed data file found")
        
        return quality
    
    def _check_model_health(self):
        """Check health of trained models"""
        health = {'models_exist': False, 'model_count': 0, 'issues': []}
        
        if os.path.exists("trained_models.joblib"):
            health['models_exist'] = True
            try:
                import joblib
                model_data = joblib.load("trained_models.joblib")
                health['model_count'] = len(model_data.get('models', {}))
            except Exception as e:
                health['issues'].append(f"Error loading models: {str(e)}")
        else:
            health['issues'].append("No trained models found")
        
        return health
    
    def _assess_system_status(self, report):
        """Assess overall system status"""
        total_issues = (len(report['data_quality'].get('issues', [])) + 
                       len(report['model_health'].get('issues', [])))
        
        if total_issues == 0:
            return 'Healthy'
        elif total_issues <= 2:
            return 'Warning'
        else:
            return 'Critical'
    
    def _generate_recommendations(self, report):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data recommendations
        if not report['data_quality'].get('data_exists'):
            recommendations.append("Run data processing pipeline to create training data")
        elif report['data_quality'].get('record_count', 0) < 20:
            recommendations.append("Collect more training data for better model performance")
        
        # Model recommendations
        if not report['model_health'].get('models_exist'):
            recommendations.append("Train initial models using available data")
        
        return recommendations if recommendations else ["System is healthy - continue monitoring"]
    
    def _generate_summary(self, results):
        """Generate performance summary with grading"""
        summary = {'overall_score': 0, 'strengths': [], 'weaknesses': [], 'grade': 'F'}
        scores = []
        
        # Regression scoring
        if 'regression' in results:
            reg = results['regression']
            dir_acc = reg.get('directional_accuracy', 0)
            
            if dir_acc >= 0.65:
                score = 90
                summary['strengths'].append(f"Excellent directional accuracy ({dir_acc:.1%})")
            elif dir_acc >= 0.55:
                score = 70
                summary['strengths'].append(f"Good directional accuracy ({dir_acc:.1%})")
            else:
                score = 50
                summary['weaknesses'].append(f"Low directional accuracy ({dir_acc:.1%})")
            
            scores.append(score)
        
        # Classification scoring
        if 'classification' in results:
            clf = results['classification']
            acc = clf.get('accuracy', 0)
            
            if acc >= 0.75:
                score = 85
                summary['strengths'].append(f"High classification accuracy ({acc:.1%})")
            elif acc >= 0.60:
                score = 70
            else:
                score = 50
                summary['weaknesses'].append(f"Low classification accuracy ({acc:.1%})")
            
            scores.append(score)
        
        # Calculate grade
        if scores:
            summary['overall_score'] = np.mean(scores)
            if summary['overall_score'] >= 85:
                summary['grade'] = 'A'
            elif summary['overall_score'] >= 75:
                summary['grade'] = 'B'
            elif summary['overall_score'] >= 65:
                summary['grade'] = 'C'
            elif summary['overall_score'] >= 55:
                summary['grade'] = 'D'
        
        return summary
    
    def _plot_regression_results(self, reg_results):
        """Create regression performance plots"""
        try:
            actual = reg_results['actual']
            predicted = reg_results['predictions']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Regression Performance', fontsize=14)
            
            # Actual vs Predicted
            axes[0, 0].scatter(actual, predicted, alpha=0.6)
            axes[0, 0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            
            # Residuals
            residuals = actual - predicted
            axes[0, 1].scatter(predicted, residuals, alpha=0.6)
            axes[0, 1].axhline(y=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('Predicted')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title('Residuals')
            
            # Residual distribution
            axes[1, 0].hist(residuals, bins=15, alpha=0.7)
            axes[1, 0].set_xlabel('Residuals')
            axes[1, 0].set_title('Residual Distribution')
            
            # Performance metrics text
            axes[1, 1].text(0.1, 0.8, f"MAE: {reg_results['mae']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f"RMSE: {reg_results['rmse']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f"R²: {reg_results['r2']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.2, f"Dir. Acc: {reg_results['directional_accuracy']:.1%}", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Metrics')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            plt.savefig('regression_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Regression plots saved")
            
        except Exception as e:
            print(f"❌ Error creating regression plots: {e}")
    
    def _plot_classification_results(self, clf_results):
        """Create classification performance plots"""
        try:
            actual = clf_results['actual']
            predicted = clf_results['predictions']
            probabilities = clf_results['probabilities']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Classification Performance', fontsize=14)
            
            # Confusion Matrix
            cm = confusion_matrix(actual, predicted)
            im = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[0, 0].set_title('Confusion Matrix')
            
            # Add labels
            classes = ['DOWN', 'UP']
            tick_marks = np.arange(len(classes))
            axes[0, 0].set_xticks(tick_marks)
            axes[0, 0].set_yticks(tick_marks)
            axes[0, 0].set_xticklabels(classes)
            axes[0, 0].set_yticklabels(classes)
            
            # Add text annotations
            for i in range(len(classes)):
                for j in range(len(classes)):
                    axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max() / 2 else "black")
            
            # Confidence distribution
            max_probs = np.max(probabilities, axis=1)
            axes[0, 1].hist(max_probs, bins=15, alpha=0.7)
            axes[0, 1].set_xlabel('Prediction Confidence')
            axes[0, 1].set_title('Confidence Distribution')
            
            # Class distribution
            unique, counts = np.unique(actual, return_counts=True)
            axes[1, 0].bar([classes[i] for i in unique], counts)
            axes[1, 0].set_title('Class Distribution')
            
            # Metrics text
            axes[1, 1].text(0.1, 0.8, f"Accuracy: {clf_results['accuracy']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.6, f"Precision: {clf_results['precision']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.4, f"Recall: {clf_results['recall']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].text(0.1, 0.2, f"F1-Score: {clf_results['f1']:.3f}", transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Metrics')
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])
            
            plt.tight_layout()
            plt.savefig('classification_performance.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Classification plots saved")
            
        except Exception as e:
            print(f"❌ Error creating classification plots: {e}")
    
    def _save_results(self, results, filename):
        """Save results to JSON file"""
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(results)
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            print(f"✅ Results saved to '{filename}'")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def _save_monitoring_report(self, report):
        """Save monitoring report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_report_{timestamp}.json"
        self._save_results(report, filename)
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays and other objects for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _print_evaluation_summary(self, results):
        """Print evaluation summary"""
        print(f"\n📊 EVALUATION SUMMARY")
        print("=" * 40)
        
        summary = results.get('summary', {})
        print(f"Overall Grade: {summary.get('grade', 'N/A')}")
        print(f"Overall Score: {summary.get('overall_score', 0):.1f}/100")
        
        if 'regression' in results:
            reg = results['regression']
            print(f"\n📈 Regression:")
            print(f"  Directional Accuracy: {reg.get('directional_accuracy', 0):.1%}")
            print(f"  R² Score: {reg.get('r2', 0):.3f}")
            print(f"  MAE: {reg.get('mae', 0):.3f}%")
        
        if 'classification' in results:
            clf = results['classification']
            print(f"\n📊 Classification:")
            print(f"  Accuracy: {clf.get('accuracy', 0):.1%}")
            print(f"  F1-Score: {clf.get('f1', 0):.3f}")
        
        for strength in summary.get('strengths', []):
            print(f"✅ {strength}")
        for weakness in summary.get('weaknesses', []):
            print(f"⚠️  {weakness}")
    
    def _print_reliability_summary(self, results):
        """Print reliability test summary"""
        print(f"\n🔧 RELIABILITY SUMMARY")
        print("=" * 40)
        
        success_rate = results.get('success_rate', 0)
        avg_time = results.get('avg_processing_time', 0)
        
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Average Time: {avg_time:.2f} seconds")
        
        if success_rate >= 0.90 and avg_time <= 5.0:
            print("✅ System reliability: EXCELLENT")
        elif success_rate >= 0.80:
            print("✅ System reliability: GOOD")
        else:
            print("⚠️  System reliability: NEEDS ATTENTION")
    
    def _print_monitoring_report(self, report):
        """Print monitoring report"""
        print(f"\n🔍 SYSTEM STATUS: {report.get('system_status', 'Unknown')}")
        print("=" * 40)
        
        data_quality = report.get('data_quality', {})
        print(f"📊 Data: {data_quality.get('record_count', 0)} records")
        
        model_health = report.get('model_health', {})
        print(f"🤖 Models: {model_health.get('model_count', 0)} available")
        
        for rec in report.get('recommendations', []):
            print(f"💡 {rec}")


# Utility functions
def quick_evaluation(pipeline=None):
    """Quick evaluation of the pipeline"""
    evaluator = PipelineEvaluator(pipeline)
    return {
        'evaluation': evaluator.evaluate_model_performance(),
        'reliability': evaluator.test_pipeline_reliability(),
        'monitoring': evaluator.generate_monitoring_report()
    }

def comprehensive_test():
    """Run comprehensive testing suite"""
    print("COMPREHENSIVE PIPELINE TESTING")
    print("="*50)
    
    try:
        from main_pipeline import KAPPipeline
        pipeline = KAPPipeline()
        
        print("Running full pipeline...")
        pipeline.run_full_pipeline(force_retrain=True)
        
        evaluator = PipelineEvaluator(pipeline)
        
        results = {
            'pipeline': pipeline,
            'evaluation': evaluator.evaluate_model_performance(),
            'reliability': evaluator.test_pipeline_reliability(),
            'monitoring': evaluator.generate_monitoring_report()
        }
        
        print("\n✅ Comprehensive testing completed!")
        return results
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None

if __name__ == "__main__":
    comprehensive_test()