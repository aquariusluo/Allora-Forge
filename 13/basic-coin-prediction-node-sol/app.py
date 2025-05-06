from flask import Flask
from model import get_inference
from config import TOKEN, TIMEFRAME, REGION, DATA_PROVIDER

app = Flask(__name__)

print(f"Loaded app.py (minimal version) with TIMEFRAME={TIMEFRAME}, TOKEN={TOKEN}")

@app.route('/inference/<token>', methods=['GET'])
def inference(token):
    try:
        log_return = get_inference(token, TIMEFRAME, REGION, DATA_PROVIDER)
        return str(log_return), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
