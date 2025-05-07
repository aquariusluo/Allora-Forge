# optimized-v27
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

MODEL_VERSION = "2025-05-07-optimized-v27"
TRAINING_DAYS = 90
print(f"[{datetime.now()}] Loaded model.py version {MODEL_VERSION} (ETHUSDT, enhanced features, Solana RPC) at {os.path.abspath(__file__)} with TIMEFRAME={TIMEFRAME}, TRAINING_DAYS={TRAINING_DAYS}")

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
    """Fetch Solana on-chain data using Solana JSON-RPC."""
    try:
        url = "https://api.mainnet-beta.solana.com"
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getRecentPerformanceSamples",
            "params": [1]
        }
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        tx_volume = data["result"][0]["numTransactions"] if data["result"] else 0
        return {
            'tx_volume': tx_volume,
            'active_addresses': 0  # Placeholder; Solana RPC doesn't provide this directly
        }
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching Solana on-chain data: {str(e)}")
        return {'tx_volume': 0, 'active_addresses': 0}

def fetch_binance_order_book(pair, region):
    """Fetch order book depth from Binance API."""
    try:
        url = f"https://api.binance.{region}/api/v3/depth?symbol={pair}&limit=10"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        bid_volume = sum(float(bid[1]) for bid in data["bids"])
        ask_volume = sum(float(ask[1]) for ask in data["asks"])
        return {
            'bid_ask_ratio': bid_volume / (ask_volume + 1e-10),
            'order_book_imbalance': (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
        }
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching Binance order book for {pair}: {str(e)}")
        return {'bid_ask_ratio': 1.0, 'order_book_imbalance': 0.0}

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
                    df = df.dropna(subset=["date"])
                    df.set_index("date", inplace=True)
                    price_df_sol = pd.concat([price_df_sol, df])
            except Exception as e:
                print(f"[{datetime.now()}] Error processing SOL file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_eth:
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
                    price_df_eth = pd.concat([price_df_eth, df])
            except Exception as e:
                print(f"[{datetime.now()}] Error processing ETH file {file}: {str(e)}")
                skipped_files.append(file)
                continue

    if price_df_btc.empty or price_df_sol.empty or price_df_eth.empty:
        print(f"[{datetime.now()}] Warning: Partial data processed, proceeding with available data.")

    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_sol = price_df_sol.rename(columns=lambda x: f"{x}_SOLUSDT")
    price_df_eth = price_df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")
    price_df = pd.concat([price_df_btc, price_df_sol, price_df_eth], axis=1)

    if TIMEFRAME != "1m":
        price_df = price_df.resample(TIMEFRAME).agg({
            f"{metric}_{pair}": "last"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for metric in ["open", "high", "low", "close", "volume"]
        })

    price_df.interpolate(method='linear', inplace=True)
    price_df.ffill(inplace=True)
    price_df.bfill(inplace=True)

    for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
        price_df[f"log_return_{pair}"] = np.log(price_df[f"close_{pair}"].shift(-1) / price_df[f"close_{pair}"])
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 4):
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)
        price_df[f"rsi_{pair}"] = calculate_rsi(price_df[f"close_{pair}"])
        price_df[f"volatility_{pair}"] = calculate_volatility(price_df[f"close_{pair}"])
        price_df[f"ma3_{pair}"] = calculate_ma(price_df[f"close_{pair}"])
        price_df[f"macd_{pair}"] = calculate_macd(price_df[f"close_{pair}"])
        price_df[f"volume_{pair}"] = price_df[f"volume_{pair}"].shift(1)
        price_df[f"bb_upper_{pair}"], price_df[f"bb_lower_{pair}"] = calculate_bollinger_bands(price_df[f"close_{pair}"])
        order_book = fetch_binance_order_book(pair, REGION)
        price_df[f"bid_ask_ratio_{pair}"] = order_book['bid_ask_ratio']
        price_df[f"order_book_imbalance_{pair}"] = order_book['order_book_imbalance']

    price_df["hour_of_day"] = price_df.index.hour

    onchain_data = fetch_solana_onchain_data()
    price_df["sol_tx_volume"] = onchain_data['tx_volume']
    price_df["sol_active_addresses"] = onchain_data['active_addresses']

    price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]
    price_df = price_df.dropna()

    if len(price_df) == 0:
        print(f"[{datetime.now()}] Error: No data remains after preprocessing. Check data alignment or NaN handling.")
        raise ValueError("Empty DataFrame after preprocessing")
    
    price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
    print(f"[{datetime.now()}] Data saved to {training_price_data_path}, rows: {len(price_df)}")

def load_frame(file_path, timeframe):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[{datetime.now()}] Training data file {file_path} does not exist.")
    
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.interpolate(method='linear', inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    features = [
        f"{metric}_{pair}_lag{lag}"
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 4)
    ] + [
        f"{feature}_{pair}"
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
        for feature in ["rsi", "volatility", "ma3", "macd", "volume", "bb_upper", "bb_lower", "bid_ask_ratio", "order_book_imbalance"]
    ] + ["hour_of_day", "sol_tx_volume", "sol_active_addresses"]

    X = df[features]
    y = df["target_SOLUSDT"]

    if len(X) == 0:
        raise ValueError("No samples available for scaling in load_frame")

    selector = SelectKBest(score_func=mutual_info_regression, k=50)
    X_selected = selector.fit_transform(X, y)
    selected_features = [features[i] for i in selector.get_support(indices=True)]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        raise ValueError("Not enough data to split into training and test sets")
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, scaler, selected_features

def weighted_rmse(y_true, y_pred, weights):
    return np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))

def weighted_mztae(y_true, y_pred, weights):
    ref_std = np.std(y_true[-100:]) if len(y_true) >= 100 else np.std(y_true)
    return np.average(np.abs((y_true - y_pred) / ref_std), weights=weights)

def train_model(timeframe, file_path=training_price_data_path):
    X_train, X_test, y_train, y_test, scaler, features = load_frame(file_path, timeframe)

    n_splits = min(5, max(2, len(X_train) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    param_grid = {
        'learning_rate': [0.005, 0.01, 0.03, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'alpha': [0, 5],
        'lambda': [1, 3]
    }
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring=make_scorer(mean_absolute_error, greater_is_better=False),
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    xgb_model = grid_search.best_estimator_

    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        learning_rate=0.01,
        max_depth=5,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        num_leaves=20,
        min_data_in_leaf=50
    )
    lgb_model.fit(X_train, y_train, feature_name=features)

    xgb_pred_train = xgb_model.predict(X_train)
    lgb_pred_train = lgb_model.predict(X_train)
    meta_X_train = np.column_stack((xgb_pred_train, lgb_pred_train))
    meta_model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.01, min_data_in_leaf=20)
    meta_model.fit(meta_X_train, y_train)

    xgb_pred_test = xgb_model.predict(X_test)
    lgb_pred_test = lgb_model.predict(X_test)
    meta_X_test = np.column_stack((xgb_pred_test, lgb_pred_test))
    final_pred = meta_model.predict(meta_X_test)

    weights = np.abs(y_test)
    train_mae = mean_absolute_error(y_train, xgb_model.predict(X_train))
    test_mae = mean_absolute_error(y_test, final_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, xgb_model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, final_pred))
    train_r2 = r2_score(y_train, xgb_model.predict(X_train))
    test_r2 = r2_score(y_test, final_pred)
    train_weighted_rmse = weighted_rmse(y_train, xgb_model.predict(X_train), np.abs(y_train))
    test_weighted_rmse = weighted_rmse(y_test, final_pred, weights)
    train_mztae = weighted_mztae(y_train, xgb_model.predict(X_train), np.abs(y_train))
    test_mztae = weighted_mztae(y_test, final_pred, weights)
    directional_accuracy = np.mean(np.sign(final_pred) == np.sign(y_test))
    correlation, p_value = pearsonr(y_test, final_pred)

    n_successes = int(directional_accuracy * len(y_test))
    binom_p_value = binomtest(n_successes, len(y_test), p=0.5, alternative='greater').pvalue
    print(f"[{datetime.now()}] Binomial Test p-value for Directional Accuracy: {binom_p_value:.4f}")

    print(f"[{datetime.now()}] Training MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
    print(f"[{datetime.now()}] Training RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
    print(f"[{datetime.now()}] Training R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
    print(f"[{datetime.now()}] Weighted RMSE: {test_weighted_rmse:.6f}, Weighted MZTAE: {test_mztae:.6f}")
    print(f"[{datetime.now()}] Directional Accuracy: {directional_accuracy:.4f}")
    print(f"[{datetime.now()}] Correlation: {correlation:.4f}, p-value: {p_value:.4f}")

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump({'xgb': xgb_model, 'lgb': lgb_model, 'meta': meta_model}, f)
    with open(scaler_file_path, "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_weighted_rmse': train_weighted_rmse,
        'test_weighted_rmse': test_weighted_rmse,
        'train_mztae': train_mztae,
        'test_mztae': test_mztae,
        'directional_accuracy': directional_accuracy,
        'correlation': correlation,
        'correlation_p_value': p_value,
        'binom_p_value': binom_p_value
    }

    return {'xgb': xgb_model, 'lgb': lgb_model, 'meta': meta_model}, scaler, metrics, features

def get_inference(token, timeframe, region, data_provider, features, cached_data=None):
    with open(model_file_path, "rb") as f:
        models = pickle.load(f)
    xgb_model = models['xgb']
    lgb_model = models['lgb']
    meta_model = models['meta']

    df = cached_data
    if df is None:
        if data_provider == "coingecko":
            df_btc = download_coingecko_current_day_data("BTC", CG_API_KEY)
            df_sol = download_coingecko_current_day_data("SOL", CG_API_KEY)
            df_eth = download_coingecko_current_day_data("ETH", CG_API_KEY)
        else:
            df_btc = download_binance_current_day_data("BTCUSDT", region)
            df_sol = download_binance_current_day_data("SOLUSDT", region)
            df_eth = download_binance_current_day_data("ETHUSDT", region)

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

        df.interpolate(method='linear', inplace=True)
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
            order_book = fetch_binance_order_book(pair, region)
            df[f"bid_ask_ratio_{pair}"] = order_book['bid_ask_ratio']
            df[f"order_book_imbalance_{pair}"] = order_book['order_book_imbalance']

        df["hour_of_day"] = df.index.hour

        onchain_data = fetch_solana_onchain_data()
        df["sol_tx_volume"] = onchain_data['tx_volume']
        df["sol_active_addresses"] = onchain_data['active_addresses']

        df = df.interpolate(method='linear').ffill().bfill().dropna()

    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    X = df[features]
    if len(X) == 0:
        print(f"[{datetime.now()}] No valid data for prediction.")
        return 0.0

    with open(scaler_file_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

    last_row = X_scaled_df.reindex(columns=features).iloc[-1:]
    xgb_pred = xgb_model.predict(last_row)
    lgb_pred = lgb_model.predict(last_row)
    meta_X = np.column_stack((xgb_pred, lgb_pred))
    log_return_pred = meta_model.predict(meta_X)[0]

    ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=SOLUSDT'
    try:
        response = requests.get(ticker_url)
        response.raise_for_status()
        latest_price = float(response.json()['price'])
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching latest price: {str(e)}")
        return 0.0

    predicted_price = latest_price * np.exp(log_return_pred)
    print(f"[{datetime.now()}] Predicted {timeframe} SOL/USD Log Return: {log_return_pred:.6f}")
    print(f"[{datetime.now()}] Latest SOL Price: {latest_price:.3f}")
    print(f"[{datetime.now()}] Predicted SOL Price in {timeframe}: {predicted_price:.3f}")
    return log_return_pred

if __name__ == "__main__":
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files_btc, files_sol, files_eth, DATA_PROVIDER)
    model, scaler, metrics, features = train_model(TIMEFRAME)
    log_return = get_inference(TOKEN, TIMEFRAME, REGION, DATA_PROVIDER, features)
