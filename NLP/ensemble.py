# ensemble.py
# Module: Ensemble LSTM + Random Forest Predictions
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
#   from ensemble import EnsembleModel
#   import pandas as pd
#
#   # Assume df_lstm contains ['date', 'pred_tomorrow_lstm', 'pred_month_lstm']
#   # and df_rf contains ['date', 'pred_tomorrow_rf']
#   ens = EnsembleModel()
#   df_ens = ens.simple_blend(df_lstm, df_rf, w_lstm=0.6, w_rf=0.4)
#   # OR for stacking:
#   ens.fit_meta_learner(df_lstm, df_rf, df_prices)
#   metas = ens.predict_meta(df_lstm, df_rf)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class EnsembleModel:
    """
    Ensemble strategies for combining LSTM and RF predictions.
    """
    def __init__(self):
        self.meta_model = None

    def simple_blend(self, df_lstm: pd.DataFrame, df_rf: pd.DataFrame, w_lstm: float=0.5, w_rf: float=0.5) -> pd.DataFrame:
        """
        Returns a DataFrame with blended predictions:
            pred_tomorrow_blend = w_lstm * pred_tomorrow_lstm + w_rf * pred_tomorrow_rf
        """
        df = pd.merge(df_lstm[['date', 'pred_tomorrow_lstm', 'pred_month_lstm']],
                      df_rf[['date', 'pred_tomorrow_rf']], on='date', how='inner')
        df['pred_tomorrow_blend'] = w_lstm * df['pred_tomorrow_lstm'] + w_rf * df['pred_tomorrow_rf']
        df['pred_month_blend'] = w_lstm * df['pred_month_lstm'] + w_rf * df['pred_tomorrow_rf']  # or month_rf if available
        return df[['date', 'pred_tomorrow_blend', 'pred_month_blend']]

    def fit_meta_learner(self, df_lstm: pd.DataFrame, df_rf: pd.DataFrame, df_prices: pd.DataFrame):
        """
        Trains a meta-learner (linear regression) on LSTM and RF predictions to predict actual returns.
        Expects:
            df_lstm: ['date', 'pred_tomorrow_lstm']
            df_rf: ['date', 'pred_tomorrow_rf']
            df_prices: ['date', 'close']
        """
        # Merge predictions and actuals
        df = pd.merge(df_lstm[['date', 'pred_tomorrow_lstm']], df_rf[['date', 'pred_tomorrow_rf']], on='date')
        df = pd.merge(df, df_prices[['date', 'close']], on='date')
        df.sort_values('date', inplace=True)
        df['actual_return'] = df['close'].shift(-1) - df['close']
        df.dropna(subset=['actual_return'], inplace=True)

        X = df[['pred_tomorrow_lstm', 'pred_tomorrow_rf']]
        y = df['actual_return']

        self.meta_model = LinearRegression()
        self.meta_model.fit(X, y)
        return self

    def predict_meta(self, df_lstm: pd.DataFrame, df_rf: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the trained meta-learner to predict returns and returns a DataFrame with ['date', 'pred_return_meta']
        """
        if self.meta_model is None:
            raise RuntimeError("Meta-learner not trained. Call fit_meta_learner first.")
        df = pd.merge(df_lstm[['date', 'pred_tomorrow_lstm']], df_rf[['date', 'pred_tomorrow_rf']], on='date')
        X = df[['pred_tomorrow_lstm', 'pred_tomorrow_rf']]
        df['pred_return_meta'] = self.meta_model.predict(X)
        return df[['date', 'pred_return_meta']]

# Example test cases
if __name__ == '__main__':
    print("EnsembleModel module loaded.")


def test_blend_shapes():
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=5)
    df_lstm = pd.DataFrame({
        'date': dates,
        'pred_tomorrow_lstm': np.arange(5),
        'pred_month_lstm': np.arange(5) * 2
    })
    df_rf = pd.DataFrame({
        'date': dates,
        'pred_tomorrow_rf': np.arange(5) * 3
    })
    ens = EnsembleModel()
    df_blend = ens.simple_blend(df_lstm, df_rf, 0.5, 0.5)
    assert 'pred_tomorrow_blend' in df_blend.columns
    assert df_blend.shape[0] == 5

def test_meta_fit_predict():
    dates = pd.date_range('2024-01-01', periods=40)
    df_lstm = pd.DataFrame({
        'date': dates,
        'pred_tomorrow_lstm': np.random.rand(40)
    })
    df_rf = pd.DataFrame({
        'date': dates,
        'pred_tomorrow_rf': np.random.rand(40)
    })
    df_prices = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100, 140, 40)
    })
    ens = EnsembleModel().fit_meta_learner(df_lstm, df_rf, df_prices)
    df_meta = ens.predict_meta(df_lstm, df_rf)
    assert 'pred_return_meta' in df_meta.columns
    assert df_meta.shape[0] <= 40
