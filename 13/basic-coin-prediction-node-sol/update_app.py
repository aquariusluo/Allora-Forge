import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def calculate_log_return(current_price, future_price):
    return np.log(future_price / current_price)

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def generate_features_sol(data):
    data_8h = data.resample("8h", on="timestamp").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    })
    features = pd.DataFrame(index=data_8h.index)
    
    # Lagged OHLC features (reduced to 3 lags)
    for col in ["open", "high", "low", "close"]:
        for i in range(1, 4):
            features[f"{col}_SOLUSDT_lag{i}"] = data_8h[col].shift(i)
    
    # Volatility (3-period standard deviation)
    features["volatility_SOLUSDT"] = data_8h["close"].rolling(window=3).std()
    
    # Moving averages (5-period and 10-period)
    features["ma5_SOLUSDT"] = data_8h["close"].rolling(window=5).mean()
    features["ma10_SOLUSDT"] = data_8h["close"].rolling(window=10).mean()
    
    # RSI
    features["rsi_SOLUSDT"] = calculate_rsi(data_8h["close"])
    
    # Volume feature
    features["volume_SOLUSDT"] = data_8h["volume"]
    
    # Hour of day
    features["hour_of_day"] = data_8h.index.hour
    
    # Target: log-return
    current = data_8h["close"]
    future = data_8h["close"].shift(-1)
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
