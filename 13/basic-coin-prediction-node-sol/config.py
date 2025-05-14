import os
from dotenv import load_dotenv

load_dotenv()

# Directory paths
data_base_path = os.path.join(os.getcwd(), "data")
model_file_path = os.path.join(data_base_path, "model.pkl")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

# Environment variables
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", 365))
TOKEN = os.getenv("TOKEN", "SOL")
REGION = os.getenv("REGION", "com")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "binance")
MODEL = os.getenv("MODEL", "lightgbm")
UPDATE_INTERVAL = os.getenv("UPDATE_INTERVAL", "3m")
CG_API_KEY = os.getenv("CG_API_KEY", "your_coingecko_api_key_here")
