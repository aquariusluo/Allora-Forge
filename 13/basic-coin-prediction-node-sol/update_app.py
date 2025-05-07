import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import TIMEFRAME

def calculate_log_return(current_price, future_price):
    return np.log(future_price / current_price)

def calculate_volatility(data, window=3):
    return data.pct_change().rolling(window=window).std()

def calculate_ma(data, window=3):
    return data.rolling(window=window).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def generate_features_sol(data):
    try:
        data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
        data_tf = data.resample(TIMEFRAME, on="timestamp").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "last"
        })
        features = pd.DataFrame(index=data_tf.index)
        
        for col in ["open", "high", "low", "close"]:
            for i in range(1, 4):
                features[f"{col}_SOLUSDT_lag{i}"] = data_tf[col].shift(i)
        
        features["volatility_SOLUSDT"] = calculate_volatility(data_tf["close"], window=3)
        features["ma3_SOLUSDT"] = calculate_ma(data_tf["close"], window=3)
        features["macd_SOLUSDT"] = calculate_macd(data_tf["close"])
        features["rsi_SOLUSDT"] = calculate_rsi(data_tf["close"])
        features["bb_upper_SOLUSDT"], features["bb_lower_SOLUSDT"] = calculate_bollinger_bands(data_tf["close"])
        features["volume_SOLUSDT"] = data_tf["volume"].shift(1)
        features["hour_of_day"] = data_tf.index.hour
        
        current = data_tf["close"]
        future = data_tf["close"].shift(-1)
        features["target_SOLUSDT"] = calculate_log_return(current, future)
        
        print(f"[{datetime.now()}] Update app log-return stats: mean={features['target_SOLUSDT'].mean():.6f}, std={features['target_SOLUSDT'].std():.6f}, min={features['target_SOLUSDT'].min():.6f}, max={features['target_SOLUSDT'].max():.6f}")
        
        features.ffill(inplace=True)
        features.bfill(inplace=True)
        features.dropna(inplace=True)
        return features
    except Exception as e:
        print(f"[{datetime.now()}] Error generating features: {str(e)}")
        return pd.DataFrame()

def save_features():
    input_path = os.getenv("SOL_SOURCE", "data/raw_sol.csv")
    output_path = os.getenv("FEATURES_PATH", "data/features_sol.csv")

    try:
        df = pd.read_csv(input_path, parse_dates=["timestamp"])
        df.sort_values("timestamp", inplace=True)
        features = generate_features_sol(df)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features.to_csv(output_path, index=False, compression=None)
        print(f"[{datetime.now()}] SOL features saved to {output_path}")
    except Exception as e:
        print(f"[{datetime.now()}] Error saving features: {str(e)}")

if __name__ == "__main__":
    save_features()
