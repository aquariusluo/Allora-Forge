import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy.stats import pearsonr, binomtest
import xgboost as xgb
import lightgbm as lgb
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, MODEL, CG_API_KEY
from datetime import datetime

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

MODEL_VERSION = "2025-05-07-optimized-v26"
TRAINING_DAYS = 90
print(f"[{datetime.now()}] Loaded model.py version {MODEL_VERSION} (ETHUSDT, enhanced features) at {os.path.abspath(__file__)} with TIMEFRAME={TIMEFRAME}, TRAINING_DAYS={TRAINING_DAYS Huntington Beach, CA 92647

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"[{datetime.now()}] Downloaded {len(files)} new files for {token}USDT")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"[{datetime.now()}] Downloaded {len(files)} new files")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def fetch_solana_onchain_data():
    """Fetch Solana on-chain data using Helius API."""
    try:
        api_key = os.getenv("HELIUS_API_KEY", "your_helius_api_key")
        url = f"https://api.helius.xyz/v0/network-stats?api-key={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            'tx_volume': data.get('total_transactions', 0),
            'active_addresses': data.get('active_addresses', 0)
        }
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching Solana on-chain data: {str(e)}")
        return {'tx_volume': 0, 'active_addresses': 0}

def calculate_rsi(data, periods=3):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=5, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_volatility(data, window=2):
    return data.pct_change().rolling(window=window).std()

def calculate_ma(data, window=2):
    return data.rolling(window=window).mean()

def calculate_macd(data, fast=4, slow=8, signal=3):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def format_data(files_btc, files_sol, files_eth, data_provider):
    print(f"[{datetime.now()}] Using TIMEFRAME={TIMEFRAME}, TRAINING_DAYS={TRAINING_DAYS}, Model Version={MODEL_VERSION}")
    if not files_btc or not files_sol or not files_eth:
        print(f"[{datetime.now()}] Warning: No files provided for BTCUSDT, SOLUSDT, or ETHUSDT, attempting to proceed with available data.")

    if data_provider == "binance":
        files_btc = sorted([f for f in files_btc if "BTCUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_sol = sorted([f for f in files_sol if "SOLUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_eth = sorted([f for f in files_eth if "ETHUSDT" in os.path.basename(f) and f.endswith(".zip")])

    price_df_btc = pd.DataFrame()
    price_df_sol = pd.DataFrame()
    price_df_eth = pd.DataFrame()
    skipped_files = []

    if data_provider == "binance":
        for file in files_btc:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"[{datetime.now()}] File not found: {zip_file_path}")
                skipped_files.append(file)
                continue
            try:
                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    df = pd.read_csv(f, header=None)
                    df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"]
                    df["date"] = pd.to_datetime(df["end_time"], unit="ms", errors='coerce', utc=True)
                    df = df.dropna(subset=["date"])
                    df.set_index("date", inplace=True)
                    price_df_btc = pd.concat([price_df_btc, df])
            except Exception as e:
                print(f"[{datetime.now()}] Error processing BTC file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_sol:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"[{datetime.now()}] File not found: {zip_file_path}")
                skipped_files.append(file)
                continue
            try:
                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    df = pd.read_csv(f, header=None)
                    df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"]
                    df["date"] = pd.to_datetime(df["end_time"], unit="ms", errors='coerce', utc=True)
                    df = df.dropna Nowhere else in the code are these files referenced or used, so we can safely remove these lines without affecting the functionality.

---

### Optimized `app.py`
Changes:
- Improved caching mechanism to handle larger datasets.
- Added error handling for data fetching to prevent crashes.
- Optimized inference endpoint to reduce latency.
- Enhanced logging for better debugging.

<xaiArtifact artifact_id="4411672b-cc56-4583-af07-d87f3d91102c" artifact_version_id="02b7afba-02e1-4b84-88f7-caf0bcb989ae" title="app.py" contentType="text/python">
import json
import os
import time
from threading import Thread
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, UPDATE_INTERVAL, CG_API_KEY
from datetime import datetime
import requests
from scipy.stats import pearsonr
import pandas as pd
from updater import download_binance_current_day_data, download_coingecko_current_day_data

app = Flask(__name__)

print(f"[{datetime.now()}] Loaded app.py (optimized for competition 13) at {os.path.abspath(__file__)} with TIMEFRAME={TIMEFRAME}, TOKEN={TOKEN}, TRAINING_DAYS={TRAINING_DAYS}")

recent_predictions = []
recent_actuals = []
model_metrics = {}
cached_features = None
cached_data = None
last_data_update = 0
cached_raw_data = None
cached_preprocessed_data = None

def parse_interval(interval):
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 3600
    else:
        return value

def fetch_solana_onchain_data():
    try:
        api_key = os.getenv("HELIUS_API_KEY", "your_helius_api_key")
        url = f"https://api.helius.xyz/v0/network-stats?api-key={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            'tx_volume': data.get('total_transactions', 0),
            'active_addresses': data.get('active_addresses', 0)
        }
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching Solana on-chain data: {str(e)}")
        return {'tx_volume': 0, 'active_addresses': 0}

def update_data_periodically():
    global cached_data, last_data_update, cached_raw_data, cached_preprocessed_data
    interval_seconds = parse_interval(UPDATE_INTERVAL)
    while True:
        try:
            update_data()
            cached_data = fetch_and_preprocess_data()
            last_data_update = children, time.time()
        except Exception as e:
            print(f"[{datetime.now()}] Error in periodic update: {str(e)}")
        time.sleep(interval_seconds)

def fetch_and_preprocess_data():
    global cached_raw_data, cached_data, cached_preprocessed_data
    print(f"[{datetime.now()}] Fetching recent data for inference...")
    
    if cached_preprocessed_data is not None and (time.time() - last_data_update) < parse_interval(UPDATE_INTERVAL):
        print(f"[{datetime.now()}] Using cached preprocessed data: {len(cached_preprocessed_data)} rows")
        return cached_preprocessed_data

    try:
        if DATA_PROVIDER == "coingecko":
            df_btc = download_coingecko_current_day_data("BTC", CG_API_KEY)
            df_sol = download_coingecko_current_day_data("SOL", CG_API_KEY)
            df_eth = download_coingecko_current_day_data("ETH", CG_API_KEY)
        else:
            df_btc = download_binance_current_day_data("BTCUSDT", REGION)
            df_sol = download_binance_current_day_data("SOLUSDT", REGION)
            df_eth = download_binance_current_day_data("ETHUSDT", REGION)
        cached_raw_data = (df_btc, df_sol, df_eth)
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching raw data: {str(e)}")
        return cached_preprocessed_data if cached_preprocessed_data is not None else pd.DataFrame()

    df_btc, df_sol, df_eth = cached_raw_data

    try:
        df_btc['date'] = pd.to_datetime(df_btc['date'], utc=True)
        df_sol['date'] = pd.to_datetime(df_sol['date'], utc=True)
        df_eth['date'] = pd.to_datetime(df_eth['date'], utc=True)
        df_btc = df_btc.sort_values('date').drop_duplicates(subset="date", keep="last")
        df_sol = df_sol.sort_values('date').drop_duplicates(subset="date", keep="last")
        df_eth = df_eth.sort_values('date').drop_duplicates(subset="date", keep="last")

        all_dates = pd.Series(list(set(df_btc['date']).union(df_sol['date'], df_eth['date']))).sort_values()
        all_dates = pd.Index(all_dates, name='date')

        df_btc = df_btc.set_index('date').reindex(all_dates, method='ffill').reset_index()
        df_sol = df_sol.set_index('date').reindex(all_dates, method='ffill').reset_index()
        df_eth = df_eth.set_index('date').reindex(all_dates, method='ffill').reset_index()

        df_btc = df_btc.set_index("date")
        df_sol = df_sol.set_index("date")
        df_eth = df_eth.set_index("date")
        df_btc = df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
        df_sol = df_sol.rename(columns=lambda x: f"{x}_SOLUSDT")
        df_eth = df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")
        df = pd.concat([df_btc, df_sol, df_eth], axis=1)

        df.ffill(inplace=True)
        df.bfill(inplace=True)

        if TIMEFRAME != "1m":
            df = df.resample(TIMEFRAME).agg({
                f"{metric}_{pair}": "last"
                for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
                for metric in ["open", "high", "low", "close", "volume"]
            })

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            for metric in ["open", "high", "low", "close"]:
                for lag in range(1, 4):
                    df[f"{metric}_{pair}_lag{lag}"] = df[f"{metric}_{pair}"].shift(lag)
            df[f"rsi_{pair}"] = calculate_rsi(df[f"close_{pair}"], periods=3)
            df[f"volatility_{pair}"] = calculate_volatility(df[f"close_{pair}"], window=2)
            df[f"ma3_{pair}"] = calculate_ma(df[f"close_{pair}"], window=2)
            df[f"macd_{pair}"] = calculate_macd(df[f"close_{pair}"], fast=4, slow=8, signal=3)
            df[f"volume_{pair}"] = df[f"volume_{pair}"].shift(1)
            df[f"bb_upper_{pair}"], df[f"bb_lower_{pair}"] = calculate_bollinger_bands(df[f"close_{pair}"], window=5)

        df["hour_of_day"] = df.index.hour
        onchain_data = fetch_solana_onchain_data()
        df["sol_tx_volume"] = onchain_data['tx_volume']
        df["sol_active_addresses"] = onchain_data['active_addresses']

        df = df.ffill().bfill().dropna()
        cached_preprocessed_data = df
        print(f"[{datetime.now()}] Preprocessed recent data: {len(df)} rows")
        return df
    except Exception as e:
        print(f"[{datetime.now()}] Error preprocessing data: {str(e)}")
        return cached_preprocessed_data if cached_preprocessed_data is not None else pd.DataFrame()

def calculate_rsi(data, periods=3):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=5, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_volatility(data, window=2):
    return data.pct_change().rolling(window=window).std()

def calculate_ma(data, window=2):
    return data.rolling(window=window).mean()

def calculate_macd(data, fast=4, slow=8, signal=3):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def update_data():
    global model_metrics, cached_features
    print(f"[{datetime.now()}] Starting data update process...")
    data_dir = os.path.join(os.getcwd(), "data", "binance")
    price_data_file = os.path.join(os.getcwd(), "data", "price_data.csv")
    model_file = model_file_path
    scaler_file = scaler_file_path
    for path in [data_dir, price_data_file, model_file, scaler_file]:
        if os.path.exists(path):
            if os.path.isdir(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
            else:
                os.remove(path)
            print(f"[{datetime.now()}] Cleared {path}")
    
    try:
        print(f"[{datetime.now()}] Downloading BTC data...")
        files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
        print(f"[{datetime.now()}] Downloading SOL data...")
        files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
        print(f"[{datetime.now()}] Downloading ETH data...")
        files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
        if not files_btc or not files_sol or not files_eth:
            print(f"[{datetime.now()}] Warning: No data files downloaded for one or more pairs.")
        print(f"[{datetime.now()}] Files downloaded - BTC: {len(files_btc)}, SOL: {len(files_sol)}, ETH: {len(files_eth)}")
        print(f"[{datetime.now()}] Formatting data...")
        format_data(files_btc, files_sol, files_eth, DATA_PROVIDER)
        print(f"[{datetime.now()}] Training model...")
        model, scaler, metrics, features = train_model(TIMEFRAME)
        model_metrics = metrics
        cached_features = features
        print(f"[{datetime.now()}] Data update and training completed.")
    except Exception as e:
        print(f"[{datetime.now()}] Update failed: {str(e)}")

@app.route("/inference/<string:token>")
def generate_inference(token):
    global cached_data, cached_features, model_metrics, last_data_update
    try:
        if not token or token.upper() != TOKEN:
            error_msg = "Token is required" if not token else f"Token {token} not supported, expected {TOKEN}"
            return Response(error_msg, status=400, mimetype='text/plain')
        if not os.path.exists(model_file_path):
            raise FileNotFoundError("Model file not found.")
        if cached_features is None or cached_data is None or (time.time() - last_data_update) > parse_interval(UPDATE_INTERVAL):
            update_data()
            cached_data = fetch_and_preprocess_data()
        
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER, cached_features, cached_data)
        
        recent_predictions.append(inference)
        if len(recent_predictions) > 100:
            recent_predictions.pop(0)
        
        metrics_log = (
            f"[{datetime.now()}] Model Metrics:\n"
            f"Training MAE: {model_metrics.get('train_mae', 0):.6f}\n"
            f"Training RMSE: {model_metrics.get('train_rmse', 0):.6f}\n"
            f"Training R²: {model_metrics.get('train_r2', 0):.6f}\n"
            f"Test MAE (log returns): {model_metrics.get('test_mae', 0):.6f}\n"
            f"Test RMSE (log returns): {model_metrics.get('test_rmse', 0):.6f}\n"
            f"Test R²: {model_metrics.get('test_r2', 0):.6f}\n"
            f"Directional Accuracy: {model_metrics.get('directional_accuracy', 0):.4f}\n"
            f"Correlation: {model_metrics.get('correlation', 0):.4f}, p-value: {model_metrics.get('correlation_p_value', 0):.4f}\n"
            f"Binomial Test p-value: {model_metrics.get('binom_p_value', 0):.4f}"
        )
        print(metrics_log)
        
        def check_actual_price():
            time.sleep(8 * 3600)
            try:
                ticker_url = f'https://api.binance.{REGION}/api/v3/ticker/price?symbol=SOLUSDT'
                response = requests.get(ticker_url, timeout=2)
                response.raise_for_status()
                new_price = float(response.json()['price'])
                ticker_url_current = f'https://api.binance.{REGION}/api/v3/ticker/price?symbol=SOLUSDT'
                response_current = requests.get(ticker_url_current, timeout=2)
                response_current.raise_for_status()
                current_price = float(response_current.json()['price'])
                actual_log_return = np.log(new_price / current_price)
                recent_actuals.append(actual_log_return)
                if len(recent_actuals) > 100:
                    recent_actuals.pop(0)
                
                if len(recent_actuals) >= 10:
                    directional_accuracy = np.mean(np.sign(recent_predictions[-len(recent_actuals):]) == np.sign(recent_actuals))
                    correlation, p_value = pearsonr(recent_predictions[-len(recent_actuals):], recent_actuals)
                    print(f"[{datetime.now()}] Runtime Directional Accuracy: {directional_accuracy:.4f}, Correlation: {correlation:.4f}, p-value: {p_value:.4f}")
            except Exception as e:
                print(f"[{datetime.now()}] Error fetching actual price: {str(e)}")
        
        Thread(target=check_actual_price).start()
        
        return Response(f"{inference:.16f}", status=200, mimetype='text/plain')
    except requests.exceptions.Timeout:
        return Response("Request timed out", status=504, mimetype='text/plain')
    except Exception as e:
        print(f"[{datetime.now()}] Inference error: {str(e)}")
        return Response(str(e), status=500, mimetype='text/plain')

@app.route("/update")
def update():
    try:
        Thread(target=update_data).start()
        return "0"
    except Exception as e:
        print(f"[{datetime.now()}] Update failed: {str(e)}")
        return "1"

if __name__ == "__main__":
    Thread(target=update_data_periodically).start()
    print(f"[{datetime.now()}] Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
