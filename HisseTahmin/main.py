from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ========== Setup App ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockRequest(BaseModel):
    ticker: str

# ========== Helper Functions ==========

def get_stock_data(ticker: str):
    start_date = "2022-09-01"
    end_date = "2025-01-01"
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    end_date_plus = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start_date, end=end_date_plus)
    return df["Close"].dropna().astype(float)

def get_fundamentals(ticker: str):
    info = yf.Ticker(ticker).info
    def rate(value, min_, max_):
        if value is None:
            return f"Veri Yok ({min_}-{max_})"
        if min_ <= value <= max_:
            return f"Makul ({min_}-{max_})"
        return f"Makul Değil ({min_}-{max_})"
    return {
        "forwardPE": {
            "value": info.get("forwardPE"),
            "rating": rate(info.get("forwardPE"), 0, 30)
        },
        "priceToBook": {
            "value": info.get("priceToBook"),
            "rating": rate(info.get("priceToBook"), 0, 3)
        },
        "returnOnEquity": {
            "value": info.get("returnOnEquity"),
            "rating": rate(info.get("returnOnEquity"), 0.1, 0.5)
        },
        "debtToEquity": {
            "value": info.get("debtToEquity"),
            "rating": rate(info.get("debtToEquity"), 0, 30)
        }
    }

def calculate_RSI(prices: pd.Series, window: int = 14) -> float:
    try:
        if len(prices) < window + 1:
            return 50.0  # Not enough data, neutral RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        RS = gain / loss
        RSI = 100 - (100 / (1 + RS))
        last_value = RSI.iloc[-1]
        return float(last_value) if pd.notna(last_value) else 50.0
    except Exception:
        return 50.0



def train_predict_model(close_prices):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices.values.reshape(-1, 1))
    
    train_size = int(len(scaled) * 0.8)
    train_data = scaled[:train_size]
    test_data = scaled[train_size - 60:]
    
    X_train, y_train = [], []
    for i in range(60, len(train_data)):
        X_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)  # Reduce for speed

    def future_prediction(days):
        input_seq = scaled[-60:].reshape(1, 60, 1)
        preds = []
        for _ in range(days):
            pred = model.predict(input_seq, verbose=0)[0][0]
            preds.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    return future_prediction

# ========== API Route ==========

@app.post("/predict")
async def predict_stock(req: StockRequest):
    print(f"ALINAN İSTEK: {req}")
    try:
        ticker = req.ticker.upper()
        close_prices = get_stock_data(ticker)
        if close_prices.empty:
            raise ValueError("Veri alınamadı.")
        
        predict_fn = train_predict_model(close_prices)
        pred_1d = predict_fn(1)[-1]
        pred_30d = predict_fn(30)[-1]

        # RSI-based adjustment
        rsi = calculate_RSI(close_prices)
        def adjust(price, period):
            if rsi < 30:
                return price * (1.03 if period == 1 else 1.05)
            elif rsi > 70:
                return price * (0.97 if period == 1 else 0.95)
            elif 30 <= rsi <= 50:
                factor = ((rsi - 30) / 20)
                return price * (1 + factor * (0.01 if period == 1 else 0.03))
            else:
                factor = ((rsi - 50) / 20)
                return price * (1 - factor * (0.01 if period == 1 else 0.03))

        return {
            "history": [
                {"date": str(date), "close": float(value) if not isinstance(value, pd.Series) else float(value.iloc[0])}
                for date, value in close_prices[-60:].items()
            ],
            "predictions": {
                "tomorrow": round(adjust(pred_1d, 1), 2),
                "oneMonth": round(adjust(pred_30d, 30), 2),
            },
            "rsi": round(rsi, 2),
            "fundamentals": get_fundamentals(ticker)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Hata: {str(e)}")

