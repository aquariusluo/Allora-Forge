from flask import Flask
from model import download_data, format_data, train_model, get_inference
from config import TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

print(f"Loaded app.py (enhanced version) with TIMEFRAME={TIMEFRAME}, TOKEN={TOKEN}, TRAINING_DAYS={TRAINING_DAYS}")

# Download data and train model on startup
try:
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    files_sol = download_data("SOL", TRAINING_DAYS, REGION, DATA_PROVIDER)
    format_data(files_btc, files_sol, DATA_PROVIDER)
    model, scaler = train_model(TIMEFRAME)
    print("Data update and training completed.")
except Exception as e:
    print(f"Error during data update or training: {str(e)}")

@app.route('/inference/<token>', methods=['GET'])
def inference(token):
    try:
        log_return = get_inference(token, TIMEFRAME, REGION, DATA_PROVIDER)
        return str(log_return), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
