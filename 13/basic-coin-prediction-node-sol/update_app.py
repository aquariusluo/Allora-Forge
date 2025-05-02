import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import TIMEFRAME

def calculate_log_return(current_price, future_price):
    return np.log(future_price / current_price)

def calculate_rsi(data, periods=5):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_features_sol(data):
    data_tf = data.resample(TIMEFRAME, on="timestamp").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })
    features = pd.DataFrame(index=data_tf.index)
    
    # Lagged OHLC features (10 lags)
    for col in ["open", "high", "low", "close"]:
        for i in range(1, 11):
            features[f"{col}_SOLUSDT_lag{i}"] = data_tf[col].shift(i)
    
    # RSI (5-period)
    features["rsi_SOLUSDT"] = calculate_rsi(data_tf["close"], periods=5)
    
    # Hour of day
    features["hour_of_day"] = data_tf.index.hour
    
    # Target: log-return
    current = data_tf["close"]
    future = data_tf["close"].shift(-1)
    features["target_SOLUSDT"] = calculate_log_return(current, future)
    
    # Forward-fill NaNs before dropping
    features.ffill(inplace=True)
    features.dropna(inplace=True)
    return features

def save_features():
    input_path = os.getenv("SOL_SOURCE", "data/raw_sol.csv")
    output_path = os.getenv("FEATURES_PATH", "data/features_sol.csv")

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)

    features = generate_features_sol(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"SOL features saved to {output_path}")

if __name__ == "__main__":
    save_features()
