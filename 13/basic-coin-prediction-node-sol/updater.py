import os
from datetime import date, timedelta, datetime
import pathlib
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json

retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=True
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount('http://', adapter)
session.mount('https://', adapter)

files = []

def download_url(url, download_path, name=None):
    try:
        global files
        if name:
            file_name = os.path.join(download_path, name)
        else:
            file_name = os.path.join(download_path, os.path.basename(url))
        dir_path = os.path.dirname(file_name)
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(file_name):
            print(f"[{datetime.now()}] File already exists, skipping: {file_name}")
            files.append(file_name)
            return
        print(f"[{datetime.now()}] Attempting to download: {url}")
        response = session.get(url, timeout=5)
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"[{datetime.now()}] Downloaded: {url} to {file_name}")
            files.append(file_name)
        else:
            print(f"[{datetime.now()}] Failed to download {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"[{datetime.now()}] Error downloading {url}: {str(e)}")

def daterange(start_date, end_date):
    days = int((end_date - start_date).days)
    print(f"[{datetime.now()}] Date range: {start_date} to {end_date}, {days} days")
    for n in range(days):
        yield start_date + timedelta(n)

def download_binance_daily_data(pair, training_days, region, download_path):
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(training_days))
    print(f"[{datetime.now()}] Downloading {pair} data from {start_date} to {end_date}")
    
    global files
    files = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        for single_date in daterange(start_date, end_date):
            url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date}.zip"
            executor.submit(download_url, url, download_path)
    
    downloaded_files = [os.path.join(download_path, f"{pair}-1m-{d}.zip") 
                        for d in daterange(start_date, end_date) 
                        if os.path.exists(os.path.join(download_path, f"{pair}-1m-{d}.zip"))]
    print(f"[{datetime.now()}] Filtered {pair} files: {downloaded_files[:5]}, total: {len(downloaded_files)}")
    return downloaded_files

def fetch_batch(pair, region, end_time, limit):
    url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}&endTime={end_time}'
    print(f"[{datetime.now()}] Fetching {pair} data batch from: {url}")
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        resp = str(response.content, 'utf-8').rstrip()
        columns = ['start_time', 'open', 'high', 'low', 'close', 'volume', 'end_time', 'volume_usd', 'n_trades', 'taker_volume', 'taker_volume_usd', 'ignore']
        df = pd.DataFrame(json.loads(resp), columns=columns)
        df['date'] = [pd.to_datetime(x+1, unit='ms', utc=True) for x in df['end_time']]
        df[["volume", "taker_volume", "open", "high", "low", "close"]] = df[["volume", "taker_volume", "open", "high", "low", "close"]].apply(pd.to_numeric)
        print(f"[{datetime.now()}] Fetched {len(df)} rows for {pair} batch")
        return df
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching {pair} data batch: {str(e)}")
        return pd.DataFrame()

def download_binance_current_day_data(pair, region):
    limit = 1000
    total_minutes = 43200  # 30 days
    requests_needed = (total_minutes + limit - 1) // limit
    dfs = []
    end_time = int(time.time() * 1000)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(requests_needed):
            batch_end_time = end_time - (i * limit * 60 * 1000)
            futures.append(executor.submit(fetch_batch, pair, region, batch_end_time, limit))
        
        for future in futures:
            df = future.result()
            if not df.empty:
                dfs.append(df)
    
    if not dfs:
        print(f"[{datetime.now()}] No data fetched for {pair}")
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs).sort_values('end_time').reset_index(drop=True)
    if len(combined_df) > total_minutes:
        combined_df = combined_df.iloc[-total_minutes:]
    print(f"[{datetime.now()}] Total {pair} live data rows fetched: {len(combined_df)}")
    return combined_df

def get_coingecko_coin_id(token):
    token_map = {
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum',
        'BERA': 'bera'
    }
    token = token.upper()
    if token in token_map:
        return token_map[token]
    else:
        raise ValueError("Unsupported token")

def download_coingecko_data(token, training_days, download_path, CG_API_KEY):
    if training_days <= 7:
        days = 7
    elif training_days <= 14:
        days = 14
    elif training_days <= 30:
        days = 30
    elif training_days <= 90:
        days = 90
    elif training_days <= 180:
        days = 180
    elif training_days <= 365:
        days = 365
    else:
        days = "max"
    print(f"[{datetime.now()}] Days: {days}")
    coin_id = get_coingecko_coin_id(token)
    print(f"[{datetime.now()}] Coin ID: {coin_id}")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}&api_key={CG_API_KEY}'
    global files
    files = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        print(f"[{datetime.now()}] Downloading data for {coin_id}")
        name = os.path.basename(url).split("?")[0].replace("/", "_") + ".json"
        executor.submit(download_url, url, download_path, name)
    return files

def download_coingecko_current_day_data(token, CG_API_KEY):
    coin_id = get_coingecko_coin_id(token)
    print(f"[{datetime.now()}] Coin ID: {coin_id}")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1&api_key={CG_API_KEY}'
    try:
        response = session.get(url, timeout=5)
        response.raise_for_status()
        resp = str(response.content, 'utf-8').rstrip()
        columns = ['timestamp', 'open', 'high', 'low', 'close']
        df = pd.DataFrame(json.loads(resp), columns=columns)
        df['date'] = [pd.to_datetime(x, unit='ms', utc=True) for x in df['timestamp']]
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
        return df.sort_values('date').reset_index(drop=True)
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching CoinGecko data for {token}: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    download_binance_daily_data("BTCUSDT", 90, "com", "data/binance")
    download_binance_daily_data("SOLUSDT", 90, "com", "data/binance")
    download_binance_daily_data("ETHUSDT", 90, "com", "data/binance")
