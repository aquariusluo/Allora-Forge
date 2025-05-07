import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

CONFIG_VERSION = "2025-05-06-optimized-v2"
print(f"[{datetime.now()}] Loaded config.py version {CONFIG_VERSION} at {os.path.abspath(__file__)} with TIMEFRAME={os.getenv('TIMEFRAME', '8h')}, TRAINING_DAYS={os.getenv('TRAINING_DAYS', '180')}")
with open(__file__, 'r') as f:
    print(f"[{datetime.now()}] config.py content (first 100 chars): {f.read(100)}...")
print(f"[{datetime.now()}] .env TIMEFRAME={os.getenv('TIMEFRAME', 'not loaded')}")

# Configuration variables
app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
model_file_path = os.path.join(data_base_path, "model_sol.pkl")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

TOKEN = os.getenv("TOKEN", default="SOL").upper()
TRAINING_DAYS = int(os.getenv("TRAINING_DAYS", default="180"))
TIMEFRAME = os.getenv("TIMEFRAME", default="8h")
MODEL = os.getenv("MODEL", default="XGBoost")
REGION = os.getenv("REGION", default="com").lower()
if REGION in ["us", "com", "usa"]:
    REGION = "com"
else:
    REGION = "com"
DATA_PROVIDER = os.getenv("DATA_PROVIDER", default="binance").lower()
CG_API_KEY = os.getenv("CG_API_KEY", default=None)
TOPIC_ID = os.getenv("TOPIC_ID", default="57")
UPDATE_INTERVAL = os.getenv("UPDATE_INTERVAL", default="3m")
PREDICTION_WINDOW = os.getenv("PREDICTION_WINDOW", default="8h")
SOL_SOURCE = os.getenv("SOL_SOURCE", default="data/raw_sol.csv")
FEATURES_PATH = os.getenv("FEATURES_PATH", default="data/features_sol.csv")
ETH_SOURCE = os.getenv("ETH_SOURCE", default="data/raw_eth.csv")
FEATURES_PATH_ETH = os.getenv("FEATURES_PATH_ETH", default="data/features_eth.csv")
