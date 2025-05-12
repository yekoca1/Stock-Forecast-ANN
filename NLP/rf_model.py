# rf_model.py
# Module: Random Forest Modeling for Price Prediction
#
# Environment:
#   - Python 3.9+
#   - Dependencies (install with pip):
#       pandas
#       numpy
#       scikit-learn
#       joblib
#
# Usage Example:
#   from rf_model import RFModel
#   import pandas as pd
#
#   # df_features: output of NLPFeatures.transform()
#   # df_prices: DataFrame with ['date', 'close']
#   rf = RFModel()
#   X_train, y_train, X_test, y_test = rf.prepare_data(df_features, df_prices)
#   rf.fit(X_train, y_train)
#   preds = rf.predict(X_test)
#   rf.evaluate(y_test, preds)
#   rf.save_model('rf_model.joblib')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class RFModel:
    """
    Random Forest Regressor for predicting next-day and next-month stock closing prices.
    """
    def __init__(self, **rf_kwargs):
        # Default hyperparameters
        self.rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            **rf_kwargs
        )
        self.best_params_ = None

    def prepare_data(self, df_features: pd.DataFrame, df_prices: pd.DataFrame):
        """
        Aligns features and price DataFrames, creates targets for next-day and next-month returns.

        Returns:
            X_train, y_train, X_test, y_test
        """
        # Merge on date
        df = pd.merge(df_features, df_prices[['date', 'close']], on='date', how='inner')
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Create targets
        df['close_tomorrow'] = df['close'].shift(-1)
        df['close_month'] = df['close'].shift(-30)
        # Drop last 30 rows without targets
        df.dropna(subset=['close_tomorrow', 'close_month'], inplace=True)

        # Features: all except date, title, url, close, targets
        X = df.drop(columns=['date', 'title', 'url', 'text', 'close', 'close_tomorrow', 'close_month'])
        # Targets: percent change
        y1 = (df['close_tomorrow'] - df['close'])/df['close']
        y2 = (df['close_month'] - df['close'])/df['close']

        # Split train/test time-series (80/20)
        split_idx = int(len(df) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = pd.DataFrame({'tomorrow': y1.iloc[:split_idx], 'month': y2.iloc[:split_idx]})
        X_test = X.iloc[split_idx:]
        y_test = pd.DataFrame({'tomorrow': y1.iloc[split_idx:], 'month': y2.iloc[split_idx:]})

        return X_train, y_train, X_test, y_test

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, param_grid=None):
        """
        Trains the Random Forest model. If param_grid provided, uses GridSearchCV.
        """
        if param_grid:
            tscv = TimeSeriesSplit(n_splits=5)
            grid = GridSearchCV(
                estimator=self.rf,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid.fit(X, y['tomorrow'])  # Example: tuning for tomorrow's target
            self.rf = grid.best_estimator_
            self.best_params_ = grid.best_params_
        else:
            self.rf.fit(X, y['tomorrow'])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts percent change for tomorrow's close.
        """
        return self.rf.predict(X)

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray):
        """
        Prints evaluation metrics (RMSE and R^2).
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    def save_model(self, path: str):
        joblib.dump(self.rf, path)

    def load_model(self, path: str):
        self.rf = joblib.load(path)
        return self

# Example test functions (require mocking or sample data)
def test_url_and_shape():
    # Placeholder: ensure methods exist
    m = RFModel()
    assert hasattr(m, 'fit') and hasattr(m, 'predict')

if __name__ == '__main__':
    print("RFModel module loaded.")
