# main.py
# Orchestrator: End-to-End Pipeline for LSTM + RF + Ensemble on KAP Disclosures
# 
# Ensure the following scripts are in the same directory:
#   - kap_scraper.py
#   - nlp_features.py
#   - rf_model.py
#   - ensemble.py
#   - lstm_model.py    # Your existing LSTM code refactored into a class named LSTMModel
#
# Install dependencies:
#   pip install requests beautifulsoup4 pandas yfinance scikit-learn pdfminer.six nltk langdetect joblib tensorflow

import yfinance as yf
import pandas as pd

from kap_scraper import KAPScraper
from nlp_features import NLPFeatures
from rf_model import RFModel
from ensemble import EnsembleModel
from lstm_model import LSTMModel


def main():
    # Parameters
    ticker = "THYAO"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    lstm_lookback = 60
    lstm_epochs = 50
    lstm_batch = 32
    rf_max_features = 500
    blend_weight_lstm = 0.6
    blend_weight_rf = 0.4

    # 1) Scrape KAP disclosures
    print("[1] Fetching KAP disclosures...")
    scraper = KAPScraper(ticker)
    df_news = scraper.fetch_announcements(start_date=start_date, end_date=end_date)

    # Check news availability
    news_available = not df_news.empty
    if not news_available:
        print("No news items found; continuing with technical model only.")
    else:
        # 2) NLP feature extraction
        print("[2] Extracting NLP features...")
        nlp = NLPFeatures(max_features=rf_max_features)
        nlp.fit(df_news['text'])
        df_feat = nlp.transform(df_news)

        # 3) Random Forest modeling
        print("[3] Training Random Forest model...")
        rf = RFModel()
        # Download price data early for RF prepare
        prices = yf.download(f"{ticker}.IS", start=start_date, end=end_date)
        df_prices = (
            prices.reset_index()[['Date','Close']]
                  .rename(columns={'Date':'date','Close':'close'})
        )
        X_train, y_train, X_test, y_test = rf.prepare_data(df_feat, df_prices)
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        df_rf_preds = pd.DataFrame({
            'date': df_prices['date'].iloc[X_test.index],
            'pred_tomorrow_rf': preds_rf
        })

    # 4) Download price data (if not already done)
    if news_available:
        df_prices_used = df_prices
    else:
        print("[2] Downloading price data from Yahoo Finance...")
        prices = yf.download(f"{ticker}.IS", start=start_date, end=end_date)
        df_prices_used = (
            prices.reset_index()[['Date','Close']]
                  .rename(columns={'Date':'date','Close':'close'})
        )

    # 5) LSTM modeling
    print("[4] Training LSTM model and generating predictions...")
    lstm = LSTMModel(
        lookback=lstm_lookback,
        epochs=lstm_epochs,
        batch_size=lstm_batch
    )
    lstm.fit(df_prices_used)
    df_lstm_preds = lstm.predict_holdout()
    # df_lstm_preds must contain ['date','pred_tomorrow_lstm','pred_month_lstm']

    # 6) Ensemble (only if RF ran)
    if news_available:
        print("[5] Blending predictions...")
        ens = EnsembleModel()
        df_blend = ens.simple_blend(df_lstm_preds, df_rf_preds,
                                     w_lstm=blend_weight_lstm,
                                     w_rf=blend_weight_rf)
        # 7) Evaluation
        print("[6] Evaluating blended predictions...")
        df_eval = pd.merge(df_blend, df_prices_used[['date','close']], on='date', how='inner')
        df_eval['actual_return'] = df_eval['close'].shift(-1) - df_eval['close']
        df_eval.dropna(subset=['actual_return'], inplace=True)
        rmse = ((df_eval['pred_tomorrow_blend'] - df_eval['actual_return'])**2).mean()**0.5
        print(f"Blended RMSE on holdout: {rmse:.4f}")
        print(df_eval.head())

        # Save RF and LSTM models
        rf.save_model('rf_model.joblib')
        lstm.save_model('lstm_model.h5')
        print("Models saved: rf_model.joblib, lstm_model.h5")
    else:
        print("Skipping RF and Ensemble. Only LSTM predictions available.")
        # Optionally, evaluate LSTM alone
        df_eval_lstm = pd.merge(df_lstm_preds, df_prices_used[['date','close']], on='date', how='inner')
        df_eval_lstm['actual_return'] = df_eval_lstm['close'].shift(-1) - df_eval_lstm['close']
        df_eval_lstm.dropna(subset=['actual_return'], inplace=True)
        rmse_lstm = ((df_eval_lstm['pred_tomorrow_lstm'] - df_eval_lstm['actual_return'])**2).mean()**0.5
        print(f"LSTM-only RMSE on holdout: {rmse_lstm:.4f}")
        print(df_eval_lstm.head())
        lstm.save_model('lstm_model.h5')
        print("Model saved: lstm_model.h5")

if __name__ == '__main__':
    main()
