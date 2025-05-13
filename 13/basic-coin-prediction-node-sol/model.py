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
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, binomtest
import lightgbm as lgb
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, MODEL, CG_API_KEY
from datetime import datetime

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")

MODEL_VERSION = "2025-05-13-optimized-v42"
print(f"[{datetime.now()}] Loaded model.py version {MODEL_VERSION} (single model: {MODEL}, {TIMEFRAME} timeframe) at {os.path.abspath(__file__)} with TIMEFRAME={TIMEFRAME}, TRAINING_DAYS={TRAINING_DAYS}")

def download_data_binance(token, training_days, region):
    try:
        files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
        print(f"[{datetime.now()}] Downloaded {len(files)} new files for {token}USDT")
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error downloading Binance data for {token}: {str(e)}")
        return []

def download_data_coingecko(token, training_days):
    try:
        files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
        print(f"[{datetime.now()}] Downloaded {len(files)} new files")
        return files
    except Exception as e:
        print(f"[{datetime.now()}] Error downloading CoinGecko data for {token}: {str(e)}")
        return []

def download_data(token, training_days, region, data_provider):
    try:
        if data_provider == "coingecko":
            return download_data_coingecko(token, int(training_days))
        elif data_provider == "binance":
            return download_data_binance(token, training_days, region)
        else:
            raise ValueError("Unsupported data provider")
    except Exception as e:
        print(f"[{datetime.now()}] Error downloading data for {token}: {str(e)}")
        return []

def calculate_rsi(data, periods=14):
    try:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating RSI: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_volatility(data, window=3):
    try:
        return data.pct_change().rolling(window=window).std()
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating volatility: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_ma(data, window=3):
    try:
        return data.rolling(window=window).mean()
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating MA: {str(e)}")
        return pd.Series(0, index=data.index)

def calculate_macd(data, fast=12, slow=26, signal=9):
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

def format_data(files_btc, files_sol, files_eth, data_provider):
    try:
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
            print(f"[{datetime.now()}] Error: Empty data for one or more pairs (BTC: {len(price_df_btc)}, SOL: {len(price_df_sol)}, ETH: {len(price_df_eth)})")
            return pd.DataFrame()

        price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
        price_df_sol = price_df_sol.rename(columns=lambda x: f"{x}_SOLUSDT")
        price_df_eth = price_df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")
        price_df = pd.concat([price_df_btc, price_df_sol, price_df_eth], axis=1)
        print(f"[{datetime.now()}] Raw concatenated DataFrame rows: {len(price_df)}")
        print(f"[{datetime.now()}] Raw columns: {list(price_df.columns)}")

        # Convert initial columns to numeric with relaxed outlier clipping
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            for metric in ["open", "high", "low", "close", "volume"]:
                price_df[f"{metric}_{pair}"] = pd.to_numeric(price_df[f"{metric}_{pair}"], errors='coerce')
                if metric in ["open", "high", "low", "close"]:
                    q_low, q_high = price_df[f"{metric}_{pair}"].quantile([0.05, 0.95])
                    price_df[f"{metric}_{pair}"] = price_df[f"{metric}_{pair}"].clip(q_low, q_high)

        # Ensure numeric dtypes before resampling
        price_df = price_df.astype({col: 'float64' for col in price_df.columns if col not in ['date']})

        if TIMEFRAME != "1m":
            price_df = price_df.resample(TIMEFRAME, closed='right', label='right').agg({
                f"{metric}_{pair}": "last"
                for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
                for metric in ["open", "high", "low", "close", "volume"]
            })
        print(f"[{datetime.now()}] After resampling rows: {len(price_df)}")

        price_df = price_df.interpolate(method='linear').ffill().bfill()

        # Feature generation
        for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
            price_df[f"close_{pair}"] = price_df[f"close_{pair}"]
            price_df[f"pct_change_{pair}"] = price_df[f"close_{pair}"].pct_change().shift(-1)
            for lag in [1, 2]:
                price_df[f"close_{pair}_lag{lag}"] = price_df[f"close_{pair}"].shift(lag)
            price_df[f"rsi_{pair}"] = calculate_rsi(price_df[f"close_{pair}"], periods=14)
            price_df[f"volatility_{pair}"] = calculate_volatility(price_df[f"close_{pair}"], window=3)
            price_df[f"ma3_{pair}"] = calculate_ma(price_df[f"close_{pair}"], window=3)
            price_df[f"macd_{pair}"] = calculate_macd(price_df[f"close_{pair}"])
            price_df[f"volume_change_{pair}"] = calculate_volume_change(price_df[f"volume_{pair}"])

        price_df["sol_btc_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_BTCUSDT")
        price_df["sol_eth_corr"] = calculate_cross_asset_correlation(price_df, "close_SOLUSDT", "close_ETHUSDT")
        price_df["sol_btc_vol_ratio"] = price_df["volatility_SOLUSDT"] / (price_df["volatility_BTCUSDT"] + 1e-10)
        price_df["sol_btc_volume_ratio"] = price_df["volume_change_SOLUSDT"] / (price_df["volume_change_BTCUSDT"] + 1e-10)
        price_df["hour_of_day"] = price_df.index.hour

        price_df["target_SOLUSDT"] = price_df["pct_change_SOLUSDT"]

        # Convert all generated features to numeric
        feature_columns = [col for col in price_df.columns if col not in ['target_SOLUSDT', 'hour_of_day']]
        for col in feature_columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

        price_df = price_df.dropna(subset=["target_SOLUSDT"])
        print(f"[{datetime.now()}] After NaN handling rows: {len(price_df)}")
        print(f"[{datetime.now()}] Features generated: {list(price_df.columns)}")
        print(f"[{datetime.now()}] NaN counts: {price_df.isna().sum().to_dict()}")
        print(f"[{datetime.now()}] Dtypes: {price_df.dtypes.to_dict()}")
        print(f"[{datetime.now()}] Feature variances: {price_df[feature_columns].var().to_dict()}")

        if len(price_df) == 0:
            print(f"[{datetime.now()}] Error: No data remains after preprocessing. Check data alignment or NaN handling.")
            return pd.DataFrame()

        price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
        print(f"[{datetime.now()}] Data saved to {training_price_data_path}, rows: {len(price_df)}")
        return price_df

    except Exception as e:
        print(f"[{datetime.now()}] Error in format_data: {str(e)}")
        return pd.DataFrame()

def load_frame(file_path, timeframe):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[{datetime.now()}] Training data file {file_path} does not exist.")
        
        df = pd.read_csv(file_path, index_col='date', parse_dates=True)
        df = df.interpolate(method='linear').ffill().bfill()

        features = [
            f"close_{pair}_lag{lag}"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for lag in [1, 2]
        ] + [
            f"{feature}_{pair}"
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
            for feature in ["rsi", "volatility", "ma3", "macd", "volume_change"]
        ] + ["hour_of_day", "sol_btc_corr", "sol_eth_corr", "sol_btc_vol_ratio", "sol_btc_volume_ratio"]

        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"[{datetime.now()}] Missing features in load_frame: {missing_features}")
            return None, None, None, None, None, None

        X = df[features]
        y = df["target_SOLUSDT"]

        if len(X) < 200:
            print(f"[{datetime.now()}] Error: Too few samples ({len(X)}) available for scaling in load_frame")
            return None, None, None, None, None, None

        selector = SelectKBest(score_func=mutual_info_regression, k=min(10, len(features)))
        selector.fit(X, y)
        scores = selector.scores_
        print(f"[{datetime.now()}] Feature scores: {list(zip(features, scores))}")
        X_selected = selector.transform(X)
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected_df)
        X_selected_df = pd.DataFrame(X_scaled, columns=selected_features, index=X.index)

        split_idx = int(len(X) * 0.8)
        if split_idx < 100:
            print(f"[{datetime.now()}] Error: Not enough data to split into training and test sets (split_idx={split_idx})")
            return None, None, None, None, None, None
        X_train, X_test = X_selected_df[:split_idx], X_selected_df[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"[{datetime.now()}] Loaded frame: {len(X_train)} training samples, {len(X_test)} test samples, features: {selected_features}")
        print(f"[{datetime.now()}] Target variance: {y.var():.6f}")
        return X_train, X_test, y_train, y_test, scaler, selected_features

    except Exception as e:
        print(f"[{datetime.now()}] Error in load_frame: {str(e)}")
        return None, None, None, None, None, None

def weighted_rmse(y_true, y_pred, weights):
    try:
        weights = np.maximum(weights, 1e-10) / np.sum(weights)
        return np.sqrt(np.average((y_true - y_pred) ** 2, weights=weights))
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating weighted RMSE: {str(e)}")
        return float('inf')

def weighted_mztae(y_true, y_pred, weights):
    try:
        ref_std = np.std(y_true[-100:]) if len(y_true) >= 100 else np.std(y_true)
        ref_std = max(ref_std, 1e-10)
        weights = np.maximum(weights, 1e-10) / np.sum(weights)
        return np.average(np.abs((y_true - y_pred) / ref_std), weights=weights)
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating weighted MZTAE: {str(e)}")
        return float('inf')

def custom_directional_loss(y_true, y_pred):
    try:
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        directional_error = np.mean(np.sign(y_true) != np.sign(y_pred))
        return rmse + 2.0 * directional_error
    except Exception as e:
        print(f"[{datetime.now()}] Error calculating custom directional loss: {str(e)}")
        return float('inf')

def train_model(timeframe, file_path=training_price_data_path):
    try:
        X_train, X_test, y_train, y_test, scaler, features = load_frame(file_path, timeframe)
        if X_train is None:
            print(f"[{datetime.now()}] Error: Failed to load frame, cannot train model")
            return None, None, {}, []

        print(f"[{datetime.now()}] Training features: {features}")

        # Baseline: Linear Regression
        baseline_model = LinearRegression()
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_rmse = weighted_rmse(y_test, baseline_pred, np.abs(y_test))
        baseline_mztae = weighted_mztae(y_test, baseline_pred, np.abs(y_test))
        print(f"[{datetime.now()}] Baseline (Linear Regression) Weighted RMSE: {baseline_rmse:.6f}, Weighted MZTAE: {baseline_mztae:.6f}")
        print(f"[{datetime.now()}] Baseline predictions (first 5): {baseline_pred[:5]}")

        # Grid search for LightGBM
        param_grid = {
            'learning_rate': [0.01, 0.05],
            'max_depth': [5, 7],
            'num_leaves': [15, 30],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
        model = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=500,
            min_child_samples=10,
            min_split_gain=0.0,
            min_child_weight=0.0001
        )
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=TimeSeriesSplit(n_splits=3),
            scoring=make_scorer(custom_directional_loss, greater_is_better=False),
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='rmse',
                        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)])
        model = grid_search.best_estimator_
        print(f"[{datetime.now()}] Best LightGBM params: {grid_search.best_params_}")

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        print(f"[{datetime.now()}] Model predictions (first 5): {pred_test[:5]}")

        # Check prediction variance
        pred_test_std = np.std(pred_test)
        if pred_test_std < 1e-10:
            print(f"[{datetime.now()}] Warning: Constant predictions detected (std: {pred_test_std:.6e}), model may be underfitting")

        weights = np.abs(y_test)
        train_mae = mean_absolute_error(y_train, pred_train)
        test_mae = mean_absolute_error(y_test, pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
        train_r2 = r2_score(y_train, pred_train)
        test_r2 = r2_score(y_test, pred_test)
        train_weighted_rmse = weighted_rmse(y_train, pred_train, np.abs(y_train))
        test_weighted_rmse = weighted_rmse(y_test, pred_test, weights)
        train_mztae = weighted_mztae(y_train, pred_train, np.abs(y_train))
        test_mztae = weighted_mztae(y_test, pred_test, weights)
        directional_accuracy = np.mean(np.sign(pred_test) == np.sign(y_test))
        
        if np.std(pred_test) < 1e-10:
            correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = pearsonr(y_test, pred_test)

        n_successes = int(directional_accuracy * len(y_test))
        binom_p_value = binomtest(n_successes, len(y_test), p=0.5, alternative='greater').pvalue
        print(f"[{datetime.now()}] Binomial Test p-value for Directional Accuracy: {binom_p_value:.4f}")

        # Calculate 95% confidence interval for directional accuracy
        n = len(y_test)
        z = 1.96  # For 95% CI
        p_hat = directional_accuracy
        ci_lower = p_hat - z * np.sqrt((p_hat * (1 - p_hat)) / n)
        ci_upper = p_hat + z * np.sqrt((p_hat * (1 - p_hat)) / n)
        print(f"[{datetime.now()}] Directional Accuracy 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        print(f"[{datetime.now()}] Training MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
        print(f"[{datetime.now()}] Training RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
        print(f"[{datetime.now()}] Training R²: {train_r2:.6f}, Test R²: {test_r2:.6f}")
        print(f"[{datetime.now()}] Weighted RMSE: {test_weighted_rmse:.6f}, Weighted MZTAE: {test_mztae:.6f}")
        weighted_rmse_improvement = 100 * (baseline_rmse - test_weighted_rmse) / max(baseline_rmse, 1e-10)
        weighted_mztae_improvement = 100 * (baseline_mztae - test_mztae) / max(baseline_mztae, 1e-10)
        print(f"[{datetime.now()}] Weighted RMSE Improvement: {weighted_rmse_improvement:.2f}%")
        print(f"[{datetime.now()}] Weighted MZTAE Improvement: {weighted_mztae_improvement:.2f}%")
        print(f"[{datetime.now()}] Directional Accuracy: {directional_accuracy:.4f}")
        print(f"[{datetime.now()}] Correlation: {correlation:.4f}, p-value: {p_value:.4f}")
        print(f"[{datetime.now()}] Feature importances: {list(zip(features, model.feature_importances_))}")

        os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_file_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"[{datetime.now()}] Model saved to {model_file_path}, scaler saved to {scaler_file_path}")

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
            'binom_p_value': binom_p_value,
            'baseline_rmse': baseline_rmse,
            'baseline_mztae': baseline_mztae,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'weighted_rmse_improvement': weighted_rmse_improvement,
            'weighted_mztae_improvement': weighted_mztae_improvement
        }

        return model, scaler, metrics, features

    except Exception as e:
        print(f"[{datetime.now()}] Error in train_model: {str(e)}")
        return None, None, {}, []

def get_inference(token, timeframe, region, data_provider, features, cached_data=None):
    try:
        if not os.path.exists(model_file_path):
            print(f"[{datetime.now()}] Error: Model file {model_file_path} not found")
            return 0.0
        with open(model_file_path, "rb") as f:
            model = pickle.load(f)

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
            print(f"[{datetime.now()}] Raw inference DataFrame rows: {len(df)}")

            # Convert relevant columns to numeric with relaxed outlier clipping
            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
                for metric in ["open", "high", "low", "close", "volume"]:
                    df[f"{metric}_{pair}"] = pd.to_numeric(df[f"{metric}_{pair}"], errors='coerce')
                    if metric in ["open", "high", "low", "close"]:
                        q_low, q_high = df[f"{metric}_{pair}"].quantile([0.05, 0.95])
                        df[f"{metric}_{pair}"] = df[f"{metric}_{pair}"].clip(q_low, q_high)

            df = df.astype({col: 'float64' for col in df.columns if col not in ['date']})
            df = df.interpolate(method='linear').ffill().bfill()

            if TIMEFRAME != "1m":
                df = df.resample(TIMEFRAME, closed='right', label='right').agg({
                    f"{metric}_{pair}": "last"
                    for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
                    for metric in ["open", "high", "low", "close", "volume"]
                })
            print(f"[{datetime.now()}] After resampling inference rows: {len(df)}")

            for pair in ["SOLUSDT", "BTCUSDT", "ETHUSDT"]:
                df[f"close_{pair}"] = df[f"close_{pair}"]
                for lag in [1, 2]:
                    df[f"close_{pair}_lag{lag}"] = df[f"close_{pair}"].shift(lag)
                df[f"rsi_{pair}"] = calculate_rsi(df[f"close_{pair}"], periods=14)
                df[f"volatility_{pair}"] = calculate_volatility(df[f"close_{pair}"], window=3)
                df[f"ma3_{pair}"] = calculate_ma(df[f"close_{pair}"], window=3)
                df[f"macd_{pair}"] = calculate_macd(df[f"close_{pair}"])
                df[f"volume_change_{pair}"] = calculate_volume_change(df[f"volume_{pair}"])

            df["sol_btc_corr"] = calculate_cross_asset_correlation(df, "close_SOLUSDT", "close_BTCUSDT")
            df["sol_eth_corr"] = calculate_cross_asset_correlation(df, "close_SOLUSDT", "close_ETHUSDT")
            df["sol_btc_vol_ratio"] = df["volatility_SOLUSDT"] / (df["volatility_BTCUSDT"] + 1e-10)
            df["sol_btc_volume_ratio"] = df["volume_change_SOLUSDT"] / (df["volume_change_BTCUSDT"] + 1e-10)
            df["hour_of_day"] = df.index.hour

            # Convert all generated features to numeric
            feature_columns = [col for col in df.columns if col != 'hour_of_day']
            for col in feature_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.interpolate(method='linear').ffill().bfill().dropna()
            print(f"[{datetime.now()}] After NaN handling inference rows: {len(df)}")
            print(f"[{datetime.now()}] Inference features generated: {list(df.columns)}")
            print(f"[{datetime.now()}] Inference NaN counts: {df.isna().sum().to_dict()}")
            print(f"[{datetime.now()}] Inference dtypes: {df.dtypes.to_dict()}")

        missing_cols = [col for col in features if col not in df.columns]
        if missing_cols:
            print(f"[{datetime.now()}] Missing feature columns: {missing_cols}")
            return 0.0
        X = df[features]
        if len(X) == 0:
            print(f"[{datetime.now()}] No valid data for prediction.")
            return 0.0

        with open(scaler_file_path, "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=X.index)

        last_row = X_scaled_df.reindex(columns=features).iloc[-1:]
        pred = model.predict(last_row)

        ticker_url = f'https://api.binance.{region}/api/v3/ticker/price?symbol=SOLUSDT'
        try:
            response = requests.get(ticker_url)
            response.raise_for_status()
            latest_price = float(response.json()['price'])
        except Exception as e:
            print(f"[{datetime.now()}] Error fetching latest price: {str(e)}")
            return 0.0

        pct_change_pred = pred[0]
        predicted_price = latest_price * (1 + pct_change_pred)
        print(f"[{datetime.now()}] Predicted {timeframe} SOL/USD Pct Change: {pct_change_pred:.6f}")
        print(f"[{datetime.now()}] Latest SOL Price: {latest_price:.3f}")
        print(f"[{datetime.now()}] Predicted SOL Price in {timeframe}: {predicted_price:.3f}")
        return pct_change_pred

    except Exception as e:
        print(f"[{datetime.now()}] Error in get_inference: {str(e)}")
        return 0.0

if __name__ == "__main__":
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_eth = download_data("ETH", TRAINING_DAYS, REGION, DATA_PROVIDER)
    df = format_data(files_btc, files_sol, files_eth, DATA_PROVIDER)
    if not df.empty:
        model, scaler, metrics, features = train_model(TIMEFRAME)
        if model is not None:
            log_return = get_inference(TOKEN, TIMEFRAME, REGION, DATA_PROVIDER, features)
        else:
            print(f"[{datetime.now()}] Error: Training failed, cannot perform inference")
    else:
        print(f"[{datetime.now()}] Error: Data formatting failed, cannot train or infer")
