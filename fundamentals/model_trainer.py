import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# 1. Load the NLP-enhanced dataset
# Assume 'nlp_enhanced_dataset.csv' was created by nlp_processor
DATA_PATH = 'nlp_enhanced_dataset.csv'

# Adjust this if your file is in a different location

def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

# 2. Prepare features and target
# Target: 'price_change_percentage'
# Drop non-numeric or identifier columns such as 'file_path', 'stock_code', 'publication_date', 'content', etc.

def prepare_data(df):
    # Copy to avoid modifying original
    data = df.copy()
    # Drop columns that aren't features
    drop_cols = ['file_path', 'stock_code', 'publication_date', 'target_trading_date', 'content', 
                 'cleaned_text', 'direction']
    for col in drop_cols:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    # Separate X and y
    y = data['price_change_percentage']
    X = data.drop(columns=['price_change_percentage'])
    return X, y

# 3. Split into training and test sets

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 4. Train baseline models

def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def train_random_forest(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def train_xgboost(X_train, y_train):
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

# 5. Evaluate model performance

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# 6. Hyperparameter tuning (example for Random Forest)

def tune_random_forest(X, y):
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

# 7. Main execution

if __name__ == '__main__':
    # Load data
    df = load_dataset()
    X, y = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train baseline models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)

    # Evaluate
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)

    print("Linear Regression Performance:", lr_metrics)
    print("Random Forest Performance:", rf_metrics)
    print("XGBoost Performance:", xgb_metrics)

    # Hyperparameter tuning example for Random Forest
    best_rf, best_params = tune_random_forest(X_train, y_train)
    tuned_rf_metrics = evaluate_model(best_rf, X_test, y_test)
    print("Tuned Random Forest Parameters:", best_params)
    print("Tuned Random Forest Performance:", tuned_rf_metrics)

    # Save models
    joblib.dump(lr_model, 'linear_regression_model.joblib')
    joblib.dump(rf_model, 'random_forest_model.joblib')
    joblib.dump(xgb_model, 'xgboost_model.joblib')
    joblib.dump(best_rf, 'random_forest_tuned_model.joblib')

    print("All models saved to disk.")
