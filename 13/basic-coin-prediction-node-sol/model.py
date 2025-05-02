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
import xgboost as xgb
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, MODEL, CG_API_KEY

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files for {token}USDT")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")

def format_data(files_btc, files_sol, data_provider):
    print(f"Raw files for BTCUSDT: {files_btc[:5]}")
    print(f"Raw files for SOLUSDT: {files_sol[:5]}")
    print(f"Files for BTCUSDT: {len(files_btc)}, Files for SOLUSDT: {len(files_sol)}")
    if not files_btc or not files_sol:
        print("Warning: No files provided for BTCUSDT or SOLUSDT, attempting to proceed with available data.")
    
    if data_provider == "binance":
        files_btc = sorted([f for f in files_btc if "BTCUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_sol = sorted([f for f in files_sol if "SOLUSDT" in os.path.basename(f) and f.endswith(".zip")])
        print(f"Filtered BTCUSDT files: {files_btc[:5]}")
        print(f"Filtered SOLUSDT files: {files_sol[:5]}")

    if len(files_btc) == 0 or len(files_sol) == 0:
        print("Warning: No valid files to process for BTCUSDT or SOLUSDT after filtering, proceeding with available data.")

    price_df_btc = pd.DataFrame()
    price_df_sol = pd.DataFrame()
    skipped_files = []

    if data_provider == "binance":
        for file in files_btc:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"File not found: {zip_file_path}")
                skipped_files.append(file)
                continue
            try:
                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    df = pd.read_csv(f, header=None)
                    df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"]
                    df["date"] = pd.to_datetime(df["end_time"], unit="us", errors='coerce')
                    df = df.dropna(subset=["date"])
                    if df["date"].max() > pd.Timestamp("2025-03-28") or df["date"].min() < pd.Timestamp("2020-01-01"):
                        raise ValueError(f"Timestamps out of expected range in {file}: min {df['date'].min()}, max {df['date'].max()}")
                    df.set_index("date", inplace=True)
                    print(f"Processed BTC file {file} with {len(df)} rows, sample dates: {df.index[:5].tolist()}")
                    price_df_btc = pd.concat([price_df_btc, df])
            except Exception as e:
                print(f"Error processing BTC file {file}: {str(e)}")
                skipped_files.append(file)
                continue

        for file in files_sol:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"File not found: {zip_file_path}")
                skipped_files.append(file)
                continue
            try:
                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    df = pd.read_csv(f, header=None)
                    df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd", "ignore"]
                    df["date"] = pd.to_datetime(df["end_time"], unit="us", errors='coerce')
                    df = df.dropna(subset=["date"])
                    df.set_index("date", inplace=True)
                    print(f"Processed SOL file {file} with {len(df)} rows, sample dates: {df.index[:5].tolist()}")
                    price_df_sol = pd.concat([price_df_sol, df])
            except Exception as e:
                print(f"Error processing SOL file {file}: {str(e)}")
                skipped_files.append(file)
                continue

    if price_df_btc.empty and price_df_sol.empty:
        print("No data processed for BTCUSDT or SOLUSDT, cannot proceed.")
        return
    elif price_df_btc.empty or price_df_sol.empty:
        print("Warning: Partial data processed (one pair missing), proceeding with available data.")

    print(f"Skipped files due to errors: {skipped_files}")
    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_sol = price_df_sol.rename(columns=lambda x: f"{x}_SOLUSDT")
    price_df = pd.concat([price_df_btc, price_df_sol], axis=1)
    print(f"Combined DataFrame rows before resampling: {len(price_df)}")

    if TIMEFRAME != "1m":
        price_df = price_df.resample(TIMEFRAME).agg({
            f"{metric}_{pair}": "last" 
            for pair in ["SOLUSDT", "BTCUSDT"]
            for metric in ["open", "high", "low", "close"]
        } | {f"volume_{pair}": "sum" for pair in ["SOLUSDT", "BTCUSDT"]})
        print(f"Rows after resampling to {TIMEFRAME}: {len(price_df)}")

    # Forward-fill NaNs before adding features
    price_df.ffill(inplace=True)

    for pair in ["SOLUSDT", "BTCUSDT"]:
        price_df[f"log_return_{pair}"] = np.log(price_df[f"close_{pair}"].shift(-1) / price_df[f"close_{pair}"])
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 4):  # 3 lags
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)
        price_df[f"volatility_{pair}"] = price_df[f"close_{pair}"].rolling(window=2).std()
        price_df[f"ma3_{pair}"] = price_df[f"close_{pair}"].rolling(window=3).mean()
        price_df[f"macd_{pair}"] = price_df[f"close_{pair}"].ewm(span=12, adjust=False).mean() - price_df[f"close_{pair}"].ewm(span=26, adjust=False).mean()
        price_df[f"volume_{pair}"] = price_df[f"volume_{pair}"]

    price_df["hour_of_day"] = price_df.index.hour
    price_df["target_SOLUSDT"] = price_df["log_return_SOLUSDT"]
    print(f"Rows before dropna: {len(price_df)}")
    print(f"NaN counts before dropna:\n{price_df.isna().sum()}")
    price_df = price_df.dropna(subset=["target_SOLUSDT"] + [f"{metric}_SOLUSDT_lag1" for metric in ["open", "high", "low", "close"]])
    print(f"Rows after dropna: {len(price_df)}")
    
    if len(price_df) == 0:
        print("No data remains after preprocessing target dropna. Filling NaNs and saving partial data.")
        price_df.fillna(0, inplace=True)
        price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Partial data saved to {training_price_data_path}")
        return

    price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Data saved to {training_price_data_path}")

def load_frame(file_path, timeframe):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file {file_path} does not exist. Run data update first.")
    
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    if df.empty:
        print("Warning: Training data file is empty, attempting to proceed with available data.")
        df = pd.DataFrame(columns=[
            f"{metric}_{pair}_lag{lag}" 
            for pair in ["SOLUSDT", "BTCUSDT"]
            for metric in ["open", "high", "low", "close"]
            for lag in range(1, 4)
        ] + [
            f"volatility_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"ma3_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"macd_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"volume_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + ["hour_of_day", "target_SOLUSDT"])
        df.loc[0] = 0
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["SOLUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 4)
        ] + [
            f"volatility_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"ma3_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"macd_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"volume_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + ["hour_of_day"]
    
    X = df[features]
    y = df["target_SOLUSDT"]
    
    print(f"Training data stats: y mean={y.mean():.6f}, y std={y.std():.6f}, rows={len(y)}")
    print(f"NaN counts in X:\n{X.isna().sum()}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        print("Warning: Not enough data to split, using all data for training.")
        split_idx = len(X)
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def preprocess_live_data(df_btc, df_sol):
    print(f"BTC raw data rows: {len(df_btc)}, columns: {df_btc.columns.tolist()}")
    print(f"SOL raw data rows: {len(df_sol)}, columns: {df_sol.columns.tolist()}")

    if "date" in df_btc.columns:
        df_btc = df_btc.drop_duplicates(subset="date", keep="last").set_index("date")
        if df_btc.index.has_duplicates:
            print(f"Warning: BTC data still has {df_btc.index.duplicated().sum()} duplicate timestamps after deduplication")
    if "date" in df_sol.columns:
        df_sol = df_sol.drop_duplicates(subset="date", keep="last").set_index("date")
        if df_sol.index.has_duplicates:
            print(f"Warning: SOL data still has {df_sol.index.duplicated().sum()} duplicate timestamps after deduplication")
    
    df_btc = df_btc.rename(columns=lambda x: f"{x}_BTCUSDT" if x != "date" else x)
    df_sol = df_sol.rename(columns=lambda x: f"{x}_SOLUSDT" if x != "date" else x)
    
    df = pd.concat([df_btc, df_sol], axis=1)
    print(f"Raw live data rows: {len(df)}")

    if TIMEFRAME != "1m":
        df = df.resample(TIMEFRAME).agg({
            f"{metric}_{pair}": "last" 
            for pair in ["SOLUSDT", "BTCUSDT"]
            for metric in ["open", "high", "low", "close"]
        } | {f"volume_{pair}": "sum" for pair in ["SOLUSDT", "BTCUSDT"]})
        print(f"Rows after resampling to {TIMEFRAME}: {len(df)}")

    # Forward-fill NaNs before adding features
    df.ffill(inplace=True)

    for pair in ["SOLUSDT", "BTCUSDT"]:
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 4):
                df[f"{metric}_{pair}_lag{lag}"] = df[f"{metric}_{pair}"].shift(lag)
        df[f"volatility_{pair}"] = df[f"close_{pair}"].rolling(window=2).std()
        df[f"ma3_{pair}"] = df[f"close_{pair}"].rolling(window=3).mean()
        df[f"macd_{pair}"] = df[f"close_{pair}"].ewm(span=12, adjust=False).mean() - df[f"close_{pair}"].ewm(span=26, adjust=False).mean()
        df[f"volume_{pair}"] = df[f"volume_{pair}"]

    df["hour_of_day"] = df.index.hour
    
    print(f"Rows after adding features: {len(df)}")
    print(f"NaN counts before dropna:\n{df.isna().sum()}")
    df = df.dropna(subset=[f"{metric}_SOLUSDT_lag1" for metric in ["open", "high", "low", "close"]])
    print(f"Live data after preprocessing rows: {len(df)}")

    if len(df) == 0:
        print("Warning: No valid data after preprocessing. Returning default prediction.")
        return np.array([[]])

    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["SOLUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 4)
        ] + [
            f"volatility_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"ma3_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"macd_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + [
            f"volume_{pair}" for pair in ["SOLUSDT", "BTCUSDT"]
        ] + ["hour_of_day"]
    
    X = df[features]
    if len(X) == 0:
        print("Warning: No valid features after preprocessing. Returning default prediction.")
        return np.array([[]])

    with open(scaler_file_path, "rb") as f:
        scaler = pickle.load(f)
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print(f"Error in scaler.transform: {str(e)}")
        return np.array([[]])
    
    return X_scaled

def train_model(timeframe, file_path=training_price_data_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found at {file_path}. Ensure data is downloaded and formatted.")
    
    X_train, X_test, y_train, y_test, scaler = load_frame(file_path, timeframe)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    n_samples = len(X_train)
    if n_samples <= 1:
        print("Warning: Too few samples for cross-validation, training basic model without GridSearchCV.")
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.01,
            max_depth=3,
            n_estimators=100,
            subsample=0.7,
            colsample_bytree=0.5,
            alpha=10,
            lambda_=10
        )
        model.fit(X_train, y_train)
        print("Basic XGBoost model trained with default parameters.")
    else:
        n_splits = 5
        print(f"Using {n_splits} splits for cross-validation with {n_samples} samples")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        print("\n🚀 Training XGBoost Model with Grid Search...")
        param_grid = {
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 5],
            'n_estimators': [100, 200],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.5, 0.7],
            'alpha': [10, 20],
            'lambda': [10, 20]
        }
        model = xgb.XGBRegressor(objective="reg:squarederror")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"\n✅ Best Hyperparameters: {grid_search.best_params_}")
    
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    print(f"Training MAE (log returns): {train_mae:.6f}")
    print(f"Training RMSE (log returns): {train_rmse:.6f}")
    print(f"Training R²: {train_r2:.6f}")

    if len(X_test) > 0:
        test_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        r2 = r2_score(y_test, test_pred)
        print(f"Test MAE (log returns): {mae:.6f}")
        print(f"Test RMSE (log returns): {rmse:.6f}")
        print(f"Test R²: {r2:.6f}")
    else:
        print("No test data available for evaluation.")
    
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_file_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Trained model saved to {model_file_path}")
    print(f"Scaler saved to {scaler_file_path}")
    
    return model, scaler

def get_inference(token, timeframe, region, data_provider):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    if data_provider == "coingecko":
        df_btc = download_coingecko_current_day_data("BTC", CG_API_KEY)
        df_sol = download_coingecko_current_day_data("SOL", CG_API_KEY)
    else:
        try:
            df_btc = download_binance_current_day_data("BTCUSDT", region)
            df_sol = download_binance_current_day_data("SOLUSDT", region)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching live data: {str(e)} - Response: {e.response.text if e.response else 'No response'}")
            return 0.0
    
    ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=SOLUSDT'
    try:
        response = requests.get(ticker_url)
        response.raise_for_status()
        latest_price = float(response.json()['price'])
    except Exception as e:
        print(f"Error fetching latest price: {str(e)}")
        return 0.0
    
    X_new = preprocess_live_data(df_btc, df_sol)
    if X_new.size == 0:
        print("No valid data for prediction. Returning default log-return.")
        return 0.0
    
    try:
        log_return_pred = loaded_model.predict(X_new[-1].reshape(1, -1))[0]
    except Exception as e:
        print(f"Error in model prediction: {str(e)}")
        return 0.0
    
    predicted_price = latest_price * np.exp(log_return_pred)
    
    print(f"Predicted 8h SOL/USD Log Return: {log_return_pred:.6f}")
    print(f"Latest SOL Price: {latest_price:.3f}")
    print(f"Predicted SOL Price in 8h: {predicted_price:.3f}")
    return log_return_pred

if __name__ == "__main__":
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files_btc, files_sol, DATA_PROVIDER)
    model, scaler = train_model(TIMEFRAME)
    log_return = get_inference(TOKEN, TIMEFRAME, REGION, DATA_PROVIDER)
