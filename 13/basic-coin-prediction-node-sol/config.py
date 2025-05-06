import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

CONFIG_VERSION = "2025-05-06-v2"
print(f"[{datetime.now()}] Loaded config.py version {CONFIG_VERSION} with TIMEFRAME={os.getenv('TIMEFRAME', '4h')}, TRAINING_DAYS={os.getenv('TRAINING_DAYS', '180')}")

# Configuration variables
data_base_path = os.getenv("DATA_BASE_PATH", "data")
model_file_path = os.path.join(data_base_path, "model_sol.pkl")
TOKEN = os.getenv("TOKEN", "SOL")
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", 180))
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
MODEL = os.getenv("MODEL", "XGBoost")
REGION = os.getenv("REGION", "com")
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "binance")
CG_API_KEY = os.getenv("CG_API_KEY", "")
TOPIC_ID = os.getenv("TOPIC_ID", "57")
UPDATE_INTERVAL = os.getenv("UPDATE_INTERVAL", "5m")
PREDICTION_WINDOW = os.getenv("PREDICTION_WINDOW", "4h")
SOL_SOURCE = os.getenv("SOL_SOURCE", "data/raw_sol.csv")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/features_sol.csv")
