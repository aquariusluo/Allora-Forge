import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import zipfile
import logging
from config import REGION, CG_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_binance_current_day_data(pair, region):
    try:
        url = f"https://api.binance.{region}/api/v3/klines"
        end_time = int(time.time() * 1000)
        start_time = end_time - (30 * 24 * 60 * 60 * 1000)  # Last 30 days
        params = {
            "symbol": pair,
            "interval": "1m",
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        all_data = []
        call_count = 0
        while start_time < end_time:
            call_count += 1
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"[{datetime.now()}] Fetched {len(data)} rows for {pair} batch, call {call_count}, start_time: {datetime.fromtimestamp(params['startTime']/1000)}")
            if not data:
                break
            all_data.extend(data)
            params["startTime"] = int(data[-1][0]) + 60000
            time.sleep(0.1)  # Avoid rate limits
        df = pd.DataFrame(all_data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignored"
        ])
        df["date"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        logger.info(f"[{datetime.now()}] Total {pair} live data rows fetched: {len(df)}")
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"[{datetime.now()}] Error fetching Binance current day data for {pair}: {str(e)}")
        return pd.DataFrame()

def download_binance_daily_data(pair, training_days, region, output_path):
    try:
        os.makedirs(output_path, exist_ok=True)
        end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=training_days)
        current_date = start_date
        files = []
        call_count = 0
        total_rows = 0
        failed_dates = []
        while current_date < end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = f"https://data.binance.vision/data/spot/daily/klines/{pair}/1m/{pair}-1m-{date_str}.zip"
            output_file = os.path.join(output_path, f"{pair}-1m-{date_str}.zip")
            call_count += 1
            attempts = 0
            max_attempts = 5
            while attempts < max_attempts:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(output_file, "wb") as f:
                            f.write(response.content)
                        with zipfile.ZipFile(output_file, "r") as zip_ref:
                            csv_file = zip_ref.namelist()[0]
                            with zip_ref.open(csv_file) as f:
                                df = pd.read_csv(f, header=None)
                                rows = len(df)
                                total_rows += rows
                                logger.info(f"[{datetime.now()}] Downloaded {pair} for {date_str}: {rows} rows, call {call_count}, attempt {attempts + 1}")
                        files.append(output_file)
                        break
                    else:
                        logger.warning(f"[{datetime.now()}] Failed to download {pair} for {date_str}: HTTP {response.status_code}, attempt {attempts + 1}, response: {response.text[:100]}")
                        attempts += 1
                        time.sleep(2)
                except requests.exceptions.RequestException as e:
                    logger.error(f"[{datetime.now()}] Error downloading {pair} for {date_str}: {str(e)}, attempt {attempts + 1}")
                    attempts += 1
                    time.sleep(2)
            if attempts == max_attempts:
                logger.error(f"[{datetime.now()}] Gave up downloading {pair} for {date_str} after {max_attempts} attempts")
                failed_dates.append(date_str)
            current_date += timedelta(days=1)
            time.sleep(0.1)  # Avoid rate limits
        logger.info(f"[{datetime.now()}] Total API calls for {pair}: {call_count}, total rows: {total_rows}, failed dates: {failed_dates}")
        return files
    except Exception as e:
        logger.error(f"[{datetime.now()}] Error downloading Binance daily data for {pair}: {str(e)}")
        return []

def download_coingecko_current_day_data(token, api_key):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{token.lower()}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": "30",
            "interval": "minute",
            "x_cg_pro_api_key": api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(window=2).max()
        df["low"] = df["close"].rolling(window=2).min()
        df["volume"] = 0  # Placeholder
        logger.info(f"[{datetime.now()}] Total {token} live data rows fetched: {len(df)}")
        return df[["date", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        logger.error(f"[{datetime.now()}] Error fetching CoinGecko current day data for {token}: {str(e)}")
        return pd.DataFrame()

def download_coingecko_data(token, training_days, output_path, api_key):
    try:
        os.makedirs(output_path, exist_ok=True)
        url = f"https://api.coingecko.com/api/v3/coins/{token.lower()}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": str(training_days),
            "interval": "daily",
            "x_cg_pro_api_key": api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open"] = df["close"].shift(1)
        df["high"] = df["close"].rolling(window=2).max()
        df["low"] = df["close"].rolling(window=2).min()
        df["volume"] = 0  # Placeholder
        output_file = os.path.join(output_path, f"{token}-daily.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"[{datetime.now()}] Downloaded CoinGecko data for {token}: {len(df)} rows")
        return [output_file]
    except Exception as e:
        logger.error(f"[{datetime.now()}] Error downloading CoinGecko data for {token}: {str(e)}")
        return []
