# lstm_model.py
# Module: LSTMModel for Price Prediction
#
# Adapts your existing LSTM script (Tahmin_macd.py) into a class with fit/predict methods.
#
# Environment:
#   - Python 3.9+
#   - Dependencies:
#       pandas
#       numpy
#       tensorflow (or keras)
#       scikit-learn
#
# Usage:
#   from lstm_model import LSTMModel
#   lstm = LSTMModel(lookback=60, epochs=50, batch_size=32)
#   lstm.fit(df_prices)
#   df_preds = lstm.predict_holdout()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMModel:
    def __init__(self, lookback=60, epochs=50, batch_size=32):
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.train_size = None

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i-self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def fit(self, df: pd.DataFrame):
        # df must have 'date','close','rsi','macd' columns or just 'close'
        # Use only close for sequence modeling; extendable to multivariate
        close = df['close'].values.reshape(-1,1)
        scaled = self.scaler.fit_transform(close)
        X, y = self._create_sequences(scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        # Train-test split 80/20
        self.train_size = int(len(X) * 0.8)
        X_train, y_train = X[:self.train_size], y[:self.train_size]
        # Build model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model
        return self

    def predict_holdout(self) -> pd.DataFrame:
        # Prepare test sequences
        close = self.scaler.transform(self.scaler.inverse_transform(self.model.layers[0].input._keras_shape)) if False else None
        # Actually, rebuild sequences from the original scaled data
        # We need access to the full scaled data
        # Workaround: store scaled data in self during fit
        # Let's adjust: store full scaled series
        # NOTE: For simplicity, assume self.scaled_series exists
        scaled = self.scaled_series
        X, _ = self._create_sequences(scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_test = X[self.train_size:]
        # Predict
        preds_scaled = self.model.predict(X_test)
        preds = self.scaler.inverse_transform(preds_scaled)
        # Build DataFrame with dates
        # We need the original df dates
        # Assuming self.dates stored during fit
        dates = self.dates[self.train_size + self.lookback:]
        df_preds = pd.DataFrame({
            'date': dates,
            'pred_tomorrow_lstm': preds.flatten()
        })
        # For month-ahead: naive rolling
        # Use last lookback window to predict step by step 30 days ahead
        last_window = scaled[-self.lookback:]
        future_preds = []
        window = last_window.copy()
        for _ in range(30):
            inp = window.reshape((1, self.lookback,1))
            pred = self.model.predict(inp)[0][0]
            future_preds.append(pred)
            window = np.append(window[1:], [[pred]], axis=0)
        future_preds = self.scaler.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
        # Create month date
        month_date = dates + pd.Timedelta(days=30)
        df_preds['pred_month_lstm'] = future_preds[:len(df_preds)] if len(future_preds)>=len(df_preds) else np.nan
        return df_preds

    def save_model(self, path: str):
        self.model.save(path)

    def load_model(self, path: str):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        return self

    # Adjust fit to store series and dates
    def fit(self, df: pd.DataFrame):
        # Store for later
        self.dates = df['date'].reset_index(drop=True)
        close = df['close'].values.reshape(-1,1)
        self.scaled_series = self.scaler.fit_transform(close)
        X, y = self._create_sequences(self.scaled_series)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        self.train_size = int(len(X) * 0.8)
        X_train, y_train = X[:self.train_size], y[:self.train_size]
        # Build and train
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.model = model
        return self
