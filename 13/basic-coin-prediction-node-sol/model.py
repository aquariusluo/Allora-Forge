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

def format_data(files_btc, files_bera, data_provider):
    print(f"Raw files for BTCUSDT: {files_btc[:5]}")
    print(f"Raw files for BERAUSDT: {files_bera[:5]}")
    print(f"Files for BTCUSDT: {len(files_btc)}, Files for BERAUSDT: {len(files_bera)}")
    if not files_btc or not files_bera:
        print("Warning: No files provided for BTCUSDT or BERAUSDT, attempting to proceed with available data.")
    
    if data_provider == "binance":
        files_btc = sorted([f for f in files_btc if "BTCUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_bera = sorted([f for f in files_bera if "BERAUSDT" in os.path.basename(f) and f.endswith(".zip")])
        print(f"Filtered BTCUSDT files: {files_btc[:5]}")
        print(f"Filtered BERAUSDT files: {files_bera[:5]}")

    if len(files_btc) == 0 or len(files_bera) == 0:
        print("Warning: No valid files to process for BTCUSDT or BERAUSDT after filtering, proceeding with available data.")

    price_df_btc = pd.DataFrame()
    price_df_bera = pd.DataFrame()
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

        for file in files_bera:
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
                    print(f"Processed BERA file {file} with {len(df)} rows, sample dates: {df.index[:5].tolist()}")
                    price_df_bera = pd.concat([price_df_bera, df])
            except Exception as e:
                print(f"Error processing BERA file {file}: {str(e)}")
                skipped_files.append(file)
                continue

    if price_df_btc.empty and price_df_bera.empty:
        print("No data processed for BTCUSDT or BERAUSDT, cannot proceed.")
        return
    elif price_df_btc.empty or price_df_bera.empty:
        print("Warning: Partial data processed (one pair missing), proceeding with available data.")

    print(f"Skipped files due to errors: {skipped_files}")
    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_bera = price_df_bera.rename(columns=lambda x: f"{x}_BERAUSDT")
    price_df = pd.concat([price_df_btc, price_df_bera], axis=1)
    print(f"Combined DataFrame rows before resampling: {len(price_df)}")

    if TIMEFRAME != "1m":
        price_df = price_df.resample(TIMEFRAME).agg({
            f"{metric}_{pair}": "last" 
            for pair in ["BERAUSDT", "BTCUSDT"]
            for metric in ["open", "high", "low", "close"]
        })
        print(f"Rows after resampling to {TIMEFRAME}: {len(price_df)}")

    for pair in ["BERAUSDT", "BTCUSDT"]:
        price_df[f"log_return_{pair}"] = np.log(price_df[f"close_{pair}"].shift(-1) / price_df[f"close_{pair}"])
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 11):
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)

    price_df["hour_of_day"] = price_df.index.hour
    price_df["target_BERAUSDT"] = price_df["log_return_BERAUSDT"]
    print(f"Rows before dropna: {len(price_df)}")
    price_df = price_df.dropna(subset=["target_BERAUSDT"])
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
            for pair in ["BERAUSDT", "BTCUSDT"]
            for metric in ["open", "high", "low", "close"]
            for lag in range(1, 11)
        ] + ["hour_of_day", "target_BERAUSDT"])
        df.loc[0] = 0
    
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["BERAUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 11)
    ] + ["hour_of_day"]
    
    X = df[features]
    y = df["target_BERAUSDT"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(len(X) * 0.8)
    if split_idx == 0:
        print("Warning: Not enough data to split, using all data for training.")
        split_idx = len(X)
    
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def preprocess_live_data(df_btc, df_bera):
    print(f"BTC raw data rows: {len(df_btc)}, columns: {df_btc.columns.tolist()}")
    print(f"BERA raw data rows: {len(df_bera)}, columns: {df_bera.columns.tolist()}")

    if "date" in df_btc.columns:
        # Remove duplicate timestamps, keeping the last entry
        df_btc = df_btc.drop_duplicates(subset="date", keep="last").set_index("date")
        if df_btc.index.has_duplicates:
            print(f"Warning: BTC data still has {df_btc.index.duplicated().sum()} duplicate timestamps after deduplication")
    if "date" in df_bera.columns:
        # Remove duplicate timestamps, keeping the last entry
        df_bera = df_bera.drop_duplicates(subset="date", keep="last").set_index("date")
        if df_bera.index.has_duplicates:
            print(f"Warning: BERA data still has {df_bera.index.duplicated().sum()} duplicate timestamps after deduplication")
    
    df_btc = df_btc.rename(columns=lambda x: f"{x}_BTCUSDT" if x != "date" else x)
    df_bera = df_bera.rename(columns=lambda x: f"{x}_BERAUSDT" if x != "date" else x)
    
    df = pd.concat([df_btc, df_bera], axis=1)
    print(f"Raw live data rows: {len(df)}")

    if TIMEFRAME != "1m":
        df = df.resample(TIMEFRAME).agg({
            f"{metric}_{pair}": "last" 
            for pair in ["BERAUSDT", "BTCUSDT"]
            for metric in ["open", "high", "low", "close"]
        })
        print(f"Rows after resampling to {TIMEFRAME}: {len(df)}")

    for pair in ["BERAUSDT", "BTCUSDT"]:
        for metric in ["open", "high", "low", "close"]:
            for lag in range(1, 11):
                df[f"{metric}_{pair}_lag{lag}"] = df[f"{metric}_{pair}"].shift(lag)

    df["hour_of_day"] = df.index.hour
    
    print(f"Rows after adding features: {len(df)}")
    df = df.dropna()
    print(f"Live data after preprocessing rows: {len(df)}")

    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["BERAUSDT", "BTCUSDT"]
        for metric in ["open", "high", "low", "close"]
        for lag in range(1, 11)
    ] + ["hour_of_day"]
    
    X = df[features]
    if len(X) == 0:
        raise ValueError("No valid data after preprocessing live data.")
    
    with open(scaler_file_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    
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
            learning_rate=0.1,
            max_depth=3,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            alpha=10,
            lambda_=1
        )
        model.fit(X_train, y_train)
        print("Basic XGBoost model trained with default parameters.")
    else:
        n_splits = min(5, max(2, n_samples - 1))
        print(f"Using {n_splits} splits for cross-validation with {n_samples} samples")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        print("\n🚀 Training XGBoost Model with Grid Search...")
        param_grid = {
            'learning_rate': [0.01, 0.02, 0.05],
            'max_depth': [2, 3],
            'n_estimators': [50, 75, 100],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.7],
            'alpha': [10, 20],
            'lambda': [1, 10]
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
        df_bera = download_coingecko_current_day_data("BERA", CG_API_KEY)
    else:
        try:
            df_btc = download_binance_current_day_data("BTCUSDT", region)
            df_bera = download_binance_current_day_data("BERAUSDT", region)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching live data: {str(e)} - Response: {e.response.text if e.response else 'No response'}")
            raise
    
    ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=BERAUSDT'
    response = requests.get(ticker_url)
    response.raise_for_status()
    latest_price = float(response.json()['price'])
    
    X_new = preprocess_live_data(df_btc, df_bera)
    log_return_pred = loaded_model.predict(X_new[-1].reshape(1, -1))[0]
    
    predicted_price = latest_price * np.exp(log_return_pred)
    
    print(f"Predicted 1h BERA/USD Log Return: {log_return_pred:.6f}")
    print(f"Latest BERA Price: {latest_price:.3f}")
    print(f"Predicted BERA Price in 1h: {predicted_price:.3f}")
    return log_return_pred

if __name__ == "__main__":
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_bera = download_data("BERA", TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files_btc, files_bera, DATA_PROVIDER)
    model, scaler = train_model(TIMEFRAME)
    log_return = get_inference(TOKEN, TIMEFRAME, REGION, DATA_PROVIDER)
