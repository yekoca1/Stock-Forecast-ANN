import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinancialMLPipeline:
    """
    Machine Learning pipeline for financial KAP report analysis
    Implements both regression (price change prediction) and classification (direction prediction)
    """
    
    def __init__(self):
        """Initialize the ML pipeline"""
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = ['price_change_percentage', 'direction']
        
        # Model configurations
        self.model_configs = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                    'normalize': [True, False] if hasattr(LinearRegression(), 'normalize') else []
                }
            },
            'random_forest_regression': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost_regression': {
                'model': xgb.XGBRegressor(random_state=42, verbosity=0),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest_classification': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'xgboost_classification': {
                'model': xgb.XGBClassifier(random_state=42, verbosity=0),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
    
    def load_and_prepare_data(self, filepath: str = "nlp_enhanced_dataset.csv") -> pd.DataFrame:
        """
        Load and prepare the enhanced dataset from Stage 2
        
        Args:
            filepath: Path to the enhanced dataset
            
        Returns:
            Prepared DataFrame
        """
        try:
            logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
            
            # Remove duplicate rows (since your current data has duplicates)
            original_len = len(df)
            df = df.drop_duplicates()
            if len(df) < original_len:
                logger.info(f"Removed {original_len - len(df)} duplicate rows")
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # Handle categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col not in ['content', 'cleaned_text', 'named_entities', 'financial_entities']:
                    df[col] = df[col].fillna('unknown')
            
            return df
            
        except FileNotFoundError:
            logger.error(f"File {filepath} not found. Please run Stage 2 first.")
            return None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and targets for modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features, regression_target, classification_target)
        """
        # Select numerical features (exclude text columns and target columns)
        exclude_columns = [
            'content', 'cleaned_text', 'named_entities', 'financial_entities',
            'file_path', 'stock_code', 'publication_date', 'target_trading_date',
            'price_change_percentage', 'price_change_absolute', 'direction',
            'target_price', 'previous_price'
        ]
        
        # Get all numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and non-feature columns
        feature_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        logger.info(f"Selected {len(feature_columns)} features for modeling")
        logger.info(f"Feature columns: {feature_columns[:10]}...")  # Show first 10
        
        # Prepare features
        X = df[feature_columns].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Prepare targets
        y_regression = df['price_change_percentage'].copy()
        y_classification = df['direction'].copy()
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        return X, y_regression, y_classification
    
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train regression models to predict price change percentage
        
        Args:
            X: Features
            y: Target (price_change_percentage)
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['regression'] = scaler
        
        results = {}
        regression_models = ['linear_regression', 'random_forest_regression', 'xgboost_regression']
        
        for model_name in regression_models:
            logger.info(f"Training {model_name}...")
            
            try:
                # Get model and parameters
                model_config = self.model_configs[model_name]
                base_model = model_config['model']
                param_grid = model_config['params']
                
                # Use scaled data for linear regression, original for tree-based models
                if 'linear' in model_name:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Grid search with cross-validation (if enough data)
                if len(X_train) >= 10 and param_grid:  # Only do grid search if enough data
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=3, scoring='neg_mean_absolute_error',
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_model, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    # Use default parameters if not enough data for grid search
                    best_model = base_model
                    best_model.fit(X_train_model, y_train)
                    best_params = "default"
                
                # Make predictions
                y_pred_train = best_model.predict(X_train_model)
                y_pred_test = best_model.predict(X_test_model)
                
                # Calculate metrics
                train_mae = mean_absolute_error(y_train, y_pred_train)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                
                # Calculate directional accuracy
                train_direction_accuracy = np.mean(np.sign(y_pred_train) == np.sign(y_train))
                test_direction_accuracy = np.mean(np.sign(y_pred_test) == np.sign(y_test))
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_direction_accuracy': train_direction_accuracy,
                    'test_direction_accuracy': test_direction_accuracy,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test,
                    'actual_train': y_train,
                    'actual_test': y_test
                }
                
                # Store the model
                self.models[model_name] = best_model
                
                logger.info(f"{model_name} - Test MAE: {test_mae:.4f}, Test R²: {test_r2:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train classification models to predict price direction
        
        Args:
            X: Features
            y: Target (direction)
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['classification'] = scaler
        
        results = {}
        classification_models = ['logistic_regression', 'random_forest_classification', 'xgboost_classification']
        
        for model_name in classification_models:
            logger.info(f"Training {model_name}...")
            
            try:
                # Get model and parameters
                model_config = self.model_configs[model_name]
                base_model = model_config['model']
                param_grid = model_config['params']
                
                # Use scaled data for logistic regression, original for tree-based models
                if 'logistic' in model_name:
                    X_train_model = X_train_scaled
                    X_test_model = X_test_scaled
                else:
                    X_train_model = X_train
                    X_test_model = X_test
                
                # Grid search with cross-validation (if enough data)
                if len(X_train) >= 10 and len(np.unique(y_train)) > 1 and param_grid:
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=3, scoring='accuracy',
                        n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_model, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    # Use default parameters if not enough data for grid search
                    best_model = base_model
                    best_model.fit(X_train_model, y_train)
                    best_params = "default"
                
                # Make predictions
                y_pred_train = best_model.predict(X_train_model)
                y_pred_test = best_model.predict(X_test_model)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                
                # Handle case where we only have one class in test set
                try:
                    train_precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
                    test_precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    train_recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
                    test_recall = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
                    train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
                    test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
                except:
                    train_precision = test_precision = train_recall = test_recall = train_f1 = test_f1 = 0
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': best_params,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_precision': train_precision,
                    'test_precision': test_precision,
                    'train_recall': train_recall,
                    'test_recall': test_recall,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'predictions_train': y_pred_train,
                    'predictions_test': y_pred_test,
                    'actual_train': y_train,
                    'actual_test': y_test
                }
                
                # Store the model
                self.models[model_name] = best_model
                
                logger.info(f"{model_name} - Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def evaluate_models(self, regression_results: Dict, classification_results: Dict) -> pd.DataFrame:
        """
        Create comprehensive model evaluation report
        
        Args:
            regression_results: Results from regression models
            classification_results: Results from classification models
            
        Returns:
            DataFrame with model comparison
        """
        evaluation_data = []
        
        # Evaluate regression models
        for model_name, results in regression_results.items():
            if 'error' not in results:
                evaluation_data.append({
                    'Model': model_name,
                    'Type': 'Regression',
                    'Test_MAE': results['test_mae'],
                    'Test_RMSE': results['test_rmse'],
                    'Test_R2': results['test_r2'],
                    'Test_Direction_Accuracy': results['test_direction_accuracy'],
                    'Best_Params': str(results['best_params'])
                })
        
        # Evaluate classification models
        for model_name, results in classification_results.items():
            if 'error' not in results:
                evaluation_data.append({
                    'Model': model_name,
                    'Type': 'Classification',
                    'Test_Accuracy': results['test_accuracy'],
                    'Test_Precision': results['test_precision'],
                    'Test_Recall': results['test_recall'],
                    'Test_F1': results['test_f1'],
                    'Best_Params': str(results['best_params'])
                })
        
        return pd.DataFrame(evaluation_data)
    
    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance from tree-based models
        
        Args:
            X: Features DataFrame
            
        Returns:
            DataFrame with feature importance
        """
        importance_data = []
        
        # Get feature importance from tree-based models
        tree_models = ['random_forest_regression', 'xgboost_regression', 
                      'random_forest_classification', 'xgboost_classification']
        
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for feature, importance in zip(self.feature_columns, importances):
                        importance_data.append({
                            'Model': model_name,
                            'Feature': feature,
                            'Importance': importance
                        })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            # Average importance across models
            avg_importance = importance_df.groupby('Feature')['Importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('Importance', ascending=False)
            return avg_importance
        else:
            return pd.DataFrame()
    
    def save_models(self, output_dir: str = "models"):
        """
        Save trained models and scalers
        
        Args:
            output_dir: Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(output_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(output_dir, f"scaler_{scaler_name}.joblib")
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {scaler_name} scaler to {scaler_path}")
        
        # Save feature columns
        feature_path = os.path.join(output_dir, "feature_columns.joblib")
        joblib.dump(self.feature_columns, feature_path)
        logger.info(f"Saved feature columns to {feature_path}")
    
    def predict_single_report(self, features: pd.Series, model_type: str = 'best') -> Dict:
        """
        Make predictions for a single report
        
        Args:
            features: Feature vector for the report
            model_type: Type of model to use ('best', 'regression', 'classification')
            
        Returns:
            Dictionary with predictions
        """
        predictions = {}
        
        # Ensure features are in correct order
        feature_vector = features[self.feature_columns].values.reshape(1, -1)
        
        try:
            # Regression predictions
            if 'xgboost_regression' in self.models:
                reg_model = self.models['xgboost_regression']
                price_change_pred = reg_model.predict(feature_vector)[0]
                predictions['price_change_percentage'] = price_change_pred
                predictions['predicted_direction'] = 1 if price_change_pred > 0 else 0
            
            # Classification predictions
            if 'xgboost_classification' in self.models:
                clf_model = self.models['xgboost_classification']
                if hasattr(clf_model, 'predict_proba'):
                    direction_proba = clf_model.predict_proba(feature_vector)[0]
                    predictions['direction_probability'] = direction_proba.tolist()
                direction_pred = clf_model.predict(feature_vector)[0]
                predictions['direction_prediction'] = direction_pred
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            predictions['error'] = str(e)
        
        return predictions


def main():
    """
    Main function to run Stage 3: Model Development
    """
    logger.info("=== STAGE 3: MODEL DEVELOPMENT ===")
    
    # Initialize ML pipeline
    ml_pipeline = FinancialMLPipeline()
    
    # Load and prepare data
    df = ml_pipeline.load_and_prepare_data()
    if df is None:
        logger.error("Could not load data. Please ensure Stage 2 has been completed.")
        return
    
    # Check if we have enough data
    if len(df) < 5:
        logger.warning(f"Only {len(df)} samples available. Results may not be reliable.")
    
    # Prepare features and targets
    X, y_regression, y_classification = ml_pipeline.prepare_features(df)
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Regression target shape: {y_regression.shape}")
    logger.info(f"Classification target shape: {y_classification.shape}")
    
    # Train regression models
    logger.info("\n" + "="*50)
    logger.info("TRAINING REGRESSION MODELS")
    logger.info("="*50)
    regression_results = ml_pipeline.train_regression_models(X, y_regression)
    
    # Train classification models
    logger.info("\n" + "="*50)
    logger.info("TRAINING CLASSIFICATION MODELS")
    logger.info("="*50)
    classification_results = ml_pipeline.train_classification_models(X, y_classification)
    
    # Evaluate models
    evaluation_df = ml_pipeline.evaluate_models(regression_results, classification_results)
    
    # Get feature importance
    importance_df = ml_pipeline.get_feature_importance(X)
    
    # Save models
    ml_pipeline.save_models()
    
    # Save results
    evaluation_df.to_csv("model_evaluation_results.csv", index=False)
    if not importance_df.empty:
        importance_df.to_csv("feature_importance_results.csv", index=False)
    
    # Print results
    print("\n" + "="*60)
    print("STAGE 3 RESULTS SUMMARY")
    print("="*60)
    
    print("\n📊 MODEL EVALUATION RESULTS:")
    print("-" * 40)
    print(evaluation_df.to_string(index=False))
    
    if not importance_df.empty:
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES:")
        print("-" * 40)
        print(importance_df.head(10).to_string(index=False))
    
    print(f"\n💾 SAVED FILES:")
    print("-" * 20)
    print("• model_evaluation_results.csv")
    print("• feature_importance_results.csv")
    print("• models/ directory with trained models")
    
    # Performance summary
    print(f"\n🎯 PERFORMANCE HIGHLIGHTS:")
    print("-" * 30)
    
    # Best regression model
    reg_models = evaluation_df[evaluation_df['Type'] == 'Regression']
    if not reg_models.empty:
        best_reg = reg_models.loc[reg_models['Test_Direction_Accuracy'].idxmax()]
        print(f"• Best Regression: {best_reg['Model']}")
        print(f"  - Direction Accuracy: {best_reg['Test_Direction_Accuracy']:.1%}")
        print(f"  - Mean Absolute Error: {best_reg['Test_MAE']:.3f}%")
    
    # Best classification model
    clf_models = evaluation_df[evaluation_df['Type'] == 'Classification']
    if not clf_models.empty:
        best_clf = clf_models.loc[clf_models['Test_Accuracy'].idxmax()]
        print(f"• Best Classification: {best_clf['Model']}")
        print(f"  - Accuracy: {best_clf['Test_Accuracy']:.1%}")
        print(f"  - F1 Score: {best_clf['Test_F1']:.3f}")
    
    print(f"\n✅ Stage 3 completed successfully!")
    print("Ready for Stage 4: Integration & Testing")
    
    return ml_pipeline, evaluation_df, importance_df


if __name__ == "__main__":
    ml_pipeline, evaluation_df, importance_df = main()