import json
import os
import time
from threading import Thread, Timer
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, UPDATE_INTERVAL, CG_API_KEY
from datetime import datetime
import requests
from scipy.stats import pearsonr
import pandas as pd
from updater import download_binance_current_day_data, download_coingecko_current_day_data

app = Flask(__name__)

print(f"[{datetime.now()}] Loaded app.py (optimized for competition 13, LightGBM, 8h timeframe) at {os.path.abspath(__file__)} with TIMEFRAME={TIMEFRAME}, TOKEN={TOKEN}, TRAINING_DAYS={TRAINING_DAYS}")

recent_predictions = []
recent_actuals = []
model_metrics = {
    'train_mae': 0.0,
    'test_mae': 0.0,
    'train_rmse': 0.0,
    'test_rmse': 0.0,
    'train_r2': 0.0,
    'test_r2': 0.0,
    'train_weighted_rmse': 0.0,
    'test_weighted_rmse': 0.0,
    'train_mztae': 0.0,
    'test_mztae': 0.0,
    'directional_accuracy': 0.0,
    'correlation': 0.0,
    'correlation_p_value': 1.0,
    'binom_p_value': 1.0,
    'baseline_rmse': 1.0,
    'baseline_mztae': 1.0
}
cached_features = None
cached_data = None
last_data_update = 0
cached_raw_data = None
cached_preprocessed_data = None
update_timeout_flag = False

def parse_interval(interval):
    try:
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        else:
            return value
    except Exception as e:
        print(f"[{datetime.now()}] Error parsing interval {interval}: {str(e)}")
        return 180  # Default to 3 minutes

def calculate_rsi(data, periods=3):
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating RSI: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volatility(data, window=2):
    try:
        return data.pct_change().rolling(window=window).std()
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating volatility: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_ma(data, window=2):
    try:
        return data.rolling(window=window).mean()
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating MA: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_macd(data, fast=4, slow=8, signal=3):
    try:
        exp1 = data.ewm(span=fast, adjust=False).mean()
        exp2 = data.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd - signal_line
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating MACD: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volume_change(data, window=1):
    try:
        return data.pct_change(window).fillna(0)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating volume change: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_cross_asset_correlation(data, pair1, pair2, window=5):
    try:
        corr = data[pair1].pct_change().rolling(window=window).corr(data[pair2].pct_change())
        return corr.fillna(0)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating cross-asset correlation: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_rsi_ratio(data_rsi1, data_rsi2):
    try:
        return data_rsi1 / (data_rsi2 + 1e-10)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating RSI ratio: {str(e)}")
        return pd.Series(0, index=data_rsi1.index)

def update_data(retries=3, backoff=2, timeout=600):
    global model_metrics, cached_features, update_timeout_flag
    for attempt in range(retries):
        try:
            print(f"[{datetime.now()}] Starting data update process (attempt {attempt+1}/{retries})...")
            data_dir = os.path.join(os.getcwd(), "data", "binance")
            price_data_file = os.path.join(os.getcwd(), "data", "price_data.csv")
            model_file = model_file_path
            scaler_file = scaler_file_path

            # Clear existing model and scaler files to force retraining
            for path in [model_file, scaler_file]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"[{datetime.now()}] Cleared {path}")

            # Clear price data
            for path in [data_dir, price_data_file]:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        for f in os.listdir(path):
                            os.remove(os.path.join(path, f))
                    else:
                        os.remove(path)
                    print(f"[{datetime.now()}] Cleared {path}")
            
            print(f"[{datetime.now()}] Downloading BTC data...")
            files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
            print(f"[{datetime.now()}] Downloading SOL data...")
            files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
            print(f"[{datetime.now()}] Downloading ETH data...")
            files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
            if not files_btc or not files_sol or not files_eth:
                print(f"[{datetime.now()}] Warning: No data files downloaded for one or more pairs.")
                raise ValueError("Data download failed")
            print(f"[{datetime.now()}] Files downloaded - BTC: {len(files_btc)}, SOL: {len(files_sol)}, ETH: {len(files_eth)}")

            print(f"[{datetime.now()}] Formatting data...")
            df = format_data(files_btc, files_sol, files_eth, DATA_PROVIDER)
            if df.empty:
                print(f"[{datetime.now()}] Error: Data formatting returned empty DataFrame")
                raise ValueError("Data formatting failed")
            
            print(f"[{datetime.now()}] Training model...")
            model, scaler, metrics, features = train_model(TIMEFRAME)
            if model is None:
                print(f"[{datetime.now()}] Error: Training failed, model is None")
                raise ValueError("Model training failed")
            
            model_metrics.update(metrics)
            cached_features = features
            print(f"[{datetime.now()}] Data update and training completed.")
            return
        except Exception as e:
            print(f"[{datetime.now()}] Update failed (attempt {attempt+1}/{retries}): {str(e)}")
            if update_timeout_flag:
                print(f"[{datetime.now()}] Update timed out")
                update_timeout_flag = False
                break
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    print(f"[{datetime.now()}] All update attempts failed.")

def update_data_periodically():
    global cached_data, last_data_update, cached_raw_data, cached_preprocessed_data
    interval_seconds = parse_interval(UPDATE_INTERVAL)
    while True:
        try:
            update_data()
            cached_data = fetch_and_preprocess_data()
            last_data_update = time.time()
        except Exception as e:
            print(f"[{datetime.now()}] Error in periodic update: {str(e)}")
        time.sleep(interval_seconds)

def fetch_and_preprocess_data():
    global cached_raw_data, cached_data, cached_preprocessed_data
    print(f"[{datetime.now()}] Fetching recent data for inference...")
    
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
    print(f"[{datetime.now()}] Raw BTC rows: {len(df_btc)}, SOL rows: {len(df_sol)}, ETH rows: {len(df_eth)}")

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
        print(f"[{datetime.now()}] Raw concatenated DataFrame rows: {len(df)}")

        # Convert initial columns to numeric
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            for metric in ["open", "high", "low", "close", "volume"]:
                df[f"{metric}_{pair}"] = pd.to_numeric(df[f"{metric}_{pair}"], errors='coerce')

        if TIMEFRAME != "1m":
            df = df.resample(TIMEFRAME, closed='right', label='right').agg({
                f"{metric}_{pair}": "last"
                for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
                for metric in ["open", "high", "low", "close", "volume"]
            })
        print(f"[{datetime.now()}] After resampling rows: {len(df)}")

        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            for lag in [1, 2, 3, 6]:
                df[f"close_{pair}_lag{lag}"] = df[f"close_{pair}"].shift(lag)
            df[f"rsi_{pair}"] = calculate_rsi(df[f"close_{pair}"], periods=3)
            df[f"volatility_{pair}"] = calculate_volatility(df[f"close_{pair}"])
            df[f"ma3_{pair}"] = calculate_ma(df[f"close_{pair}"], window=2)
            df[f"macd_{pair}"] = calculate_macd(df[f"close_{pair}"])
            df[f"volume_change_{pair}"] = calculate_volume_change(df[f"volume_{pair}"])

        df["sol_btc_corr"] = calculate_cross_asset_correlation(df, "close_SOLUSDT", "close_BTCUSDT")
        df["sol_btc_vol_ratio"] = df["volatility_SOLUSDT"] / (df["volatility_BTCUSDT"] + 1e-10)
        df["sol_btc_volume_ratio"] = df["volume_change_SOLUSDT"] / (df["volume_change_BTCUSDT"] + 1e-10)
        df["sol_btc_rsi_ratio"] = calculate_rsi_ratio(df["rsi_SOLUSDT"], df["rsi_BTCUSDT"])
        df["hour_of_day"] = df.index.hour

        # Convert all generated features to numeric
        feature_columns = [col for col in df.columns if col != 'hour_of_day']
        for col in feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.infer_objects(copy=False).interpolate(method='linear').ffill().bfill().dropna()
        print(f"[{datetime.now()}] After NaN handling rows: {len(df)}")
        print(f"[{datetime.now()}] App inference features generated: {list(df.columns)}")
        print(f"[{datetime.now()}] App inference NaN counts: {df.isna().sum().to_dict()}")
        print(f"[{datetime.now()}] App inference dtypes: {df.dtypes.to_dict()}")

        cached_preprocessed_data = df
        return df
    except Exception as e:
        print(f"[{datetime.now()}] Error preprocessing data: {str(e)}")
        cached_preprocessed_data = None
        return pd.DataFrame()

@app.route("/inference/<string:token>")
def generate_inference(token):
    global cached_data, cached_features, model_metrics, last_data_update
    try:
        if not token or token.upper() != TOKEN:
            error_msg = "Token is required" if not token else f"Token {token} not supported, expected {TOKEN}"
            return Response(error_msg, status=400, mimetype='text/plain')
        
        if not os.path.exists(model_file_path):
            print(f"[{datetime.now()}] Model file {model_file_path} not found, attempting to update data...")
            update_data()
            if not os.path.exists(model_file_path):
                return Response("Model file not found after update attempts", status=500, mimetype='text/plain')
        
        if cached_features is None or cached_data is None or (time.time() - last_data_update) > parse_interval(UPDATE_INTERVAL):
            cached_data = fetch_and_preprocess_data()
            if cached_data.empty:
                return Response("Failed to preprocess data", status=500, mimetype='text/plain')
        
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER, cached_features, cached_data)
        if inference == 0.0:
            print(f"[{datetime.now()}] Warning: Inference returned 0.0, possible model or data issue")
            update_data()  # Retry training if inference fails
            inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER, cached_features, cached_data)
            if inference == 0.0:
                return Response("Inference failed after retry", status=500, mimetype='text/plain')
        
        recent_predictions.append(inference)
        if len(recent_predictions) > 100:
            recent_predictions.pop(0)
        
        baseline_rmse = model_metrics.get('baseline_rmse', 1.0)
        baseline_mztae = model_metrics.get('baseline_mztae', 1.0)
        test_weighted_rmse = model_metrics.get('test_weighted_rmse', float('inf'))
        test_mztae = model_metrics.get('test_mztae', float('inf'))
        rmse_improvement = 100 * (baseline_rmse - test_weighted_rmse) / baseline_rmse if baseline_rmse != 0 else 0.0
        mztae_improvement = 100 * (baseline_mztae - test_mztae) / baseline_mztae if baseline_mztae != 0 else 0.0

        metrics_log = (
            f"[{datetime.now()}] Model Metrics:\n"
            f"Training MAE: {model_metrics.get('train_mae', 0):.6f}\n"
            f"Training RMSE: {model_metrics.get('train_rmse', 0):.6f}\n"
            f"Training R²: {model_metrics.get('train_r2', 0):.6f}\n"
            f"Test MAE (log returns): {model_metrics.get('test_mae', 0):.6f}\n"
            f"Test RMSE (log returns): {model_metrics.get('test_rmse', 0):.6f}\n"
            f"Test R²: {model_metrics.get('test_r2', 0):.6f}\n"
            f"Directional Accuracy: {model_metrics.get('directional_accuracy', 0):.4f}\n"
            f"Correlation: {model_metrics.get('correlation', 0):.4f}, p-value: {model_metrics.get('correlation_p_value', 1):.4f}\n"
            f"Binomial Test p-value: {model_metrics.get('binom_p_value', 1):.4f}\n"
            f"Weighted RMSE Improvement: {rmse_improvement:.2f}%\n"
            f"Weighted MZTAE Improvement: {mztae_improvement:.2f}%"
        )
        print(metrics_log)
        
        def check_actual_price():
            try:
                time.sleep(8 * 3600)
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
        print(f"[{datetime.now()}] Inference timed out")
        return Response("Request timed out", status=504, mimetype='text/plain')
    except Exception as e:
        print(f"[{datetime.now()}] Inference error: {str(e)}")
        return Response(f"Inference failed: {str(e)}", status=500, mimetype='text/plain')

@app.route("/update")
def update():
    try:
        Thread(target=update_data).start()
        return "0"
    except Exception as e:
        print(f"[{datetime.now()}] Update failed: {str(e)}")
        return "1"

if __name__ == "__main__":
    # Clear model file on startup to ensure fresh training
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
        print(f"[{datetime.now()}] Cleared model file {model_file_path} to force retraining")
    if os.path.exists(scaler_file_path):
        os.remove(scaler_file_path)
        print(f"[{datetime.now()}] Cleared scaler file {scaler_file_path} to force retraining")
    
    # Run update_data synchronously on startup
    print(f"[{datetime.now()}] Running initial data update...")
    update_data()
    
    Thread(target=update_data_periodically).start()
    print(f"[{datetime.now()}] Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
