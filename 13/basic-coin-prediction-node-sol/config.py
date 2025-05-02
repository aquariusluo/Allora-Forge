import os
from dotenv import load_dotenv

load_dotenv()

app_base_path = os.getenv("APP_BASE_PATH", default=os.getcwd())
data_base_path = os.path.join(app_base_path, "data")
model_file_path = os.path.join(data_base_path, "model_bera.pkl")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

TOKEN = os.getenv("TOKEN", default="BERA").upper()
TRAINING_DAYS = os.getenv("TRAINING_DAYS", default="180")
TIMEFRAME = os.getenv("TIMEFRAME", default="1h")
MODEL = os.getenv("MODEL", default="XGBoost")
REGION = os.getenv("REGION", default="com").lower()
if REGION in ["us", "com", "usa"]:
    REGION = "com"
else:
    REGION = "com"
DATA_PROVIDER = os.getenv("DATA_PROVIDER", default="binance").lower()
CG_API_KEY = os.getenv("CG_API_KEY", default=None)
