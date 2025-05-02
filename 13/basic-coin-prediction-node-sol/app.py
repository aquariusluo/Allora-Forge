import json
import os
import time
from threading import Thread
from flask import Flask, Response
from model import download_data, format_data, train_model, get_inference
from config import model_file_path, scaler_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER

app = Flask(__name__)

def update_data():
    print("Starting data update process in background...")
    data_dir = os.path.join(os.getcwd(), "data", "binance")
    price_data_file = os.path.join(os.getcwd(), "data", "price_data.csv")
    model_file = model_file_path
    scaler_file = scaler_file_path
    for path in [data_dir, price_data_file, model_file, scaler_file]:
        if os.path.exists(path):
            if os.path.isdir(path):
                for f in os.listdir(path):
                    os.remove(os.path.join(path, f))
            else:
                os.remove(path)
            print(f"Cleared {path}")
    
    print("Downloading BTC data...")
    files_btc = download_data("BTC", TRAINING_DAYS, REGION, DATA_PROVIDER)
    print("Downloading BERA data...")
    files_bera = download_data("BERA", TRAINING_DAYS, REGION, DATA_PROVIDER)
    if not files_btc or not files_bera:
        print("Warning: No data files downloaded for one or both pairs. Attempting to proceed with available data.")
    print(f"Files downloaded - BTC: {len(files_btc)}, BERA: {len(files_bera)}")
    print("Formatting data...")
    format_data(files_btc, files_bera, DATA_PROVIDER)
    print("Training model...")
    train_model(TIMEFRAME)
    print("Data update and training completed.")

@app.route("/inference/<string:token>")
def generate_inference(token):
    if not token or token.upper() != TOKEN:
        error_msg = "Token is required" if not token else f"Token {token} not supported, expected {TOKEN}"
        return Response(error_msg, status=400, mimetype='text/plain')
    try:
        if not os.path.exists(model_file_path):
            raise FileNotFoundError("Model file not found. Please wait for initial training to complete.")
        inference = get_inference(token.upper(), TIMEFRAME, REGION, DATA_PROVIDER)
        return Response(f"{inference:.16f}", status=200, mimetype='text/plain')
    except Exception as e:
        return Response(str(e), status=500, mimetype='text/plain')

@app.route("/update")
def update():
    try:
        Thread(target=update_data).start()  # Run update in background
        return "0"
    except Exception as e:
        print(f"Update failed: {str(e)}")
        return "1"

if __name__ == "__main__":
    # Start initial data update in a separate thread
    Thread(target=update_data).start()
    # Start Flask server immediately without waiting for training
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=8000)
