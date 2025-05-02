import os
from datetime import date, timedelta
import pathlib
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import json

retry_strategy = Retry(
    total=4,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
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
            print(f"File already exists, skipping: {file_name}")
            files.append(file_name)
            return
        print(f"Attempting to download: {url}")
        response = session.get(url)
        if response.status_code == 404:
            print(f"File does not exist: {url}")
        elif response.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {url} to {file_name}")
            files.append(file_name)
        else:
            print(f"Failed to download {url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

def daterange(start_date, end_date):
    days = int((end_date - start_date).days)
    print(f"Date range: {start_date} to {end_date}, {days} days")
    for n in range(days):
        yield start_date + timedelta(n)

def download_binance_daily_data(pair, training_days, region, download_path):
    base_url = f"https://data.binance.vision/data/spot/daily/klines"
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=int(training_days))
    print(f"Downloading {pair} data from {start_date} to {end_date}")
    
    global files
    files = []

    with ThreadPoolExecutor() as executor:
        for single_date in daterange(start_date, end_date):
            url = f"{base_url}/{pair}/1m/{pair}-1m-{single_date}.zip"
            executor.submit(download_url, url, download_path)
    
    downloaded_files = [os.path.join(download_path, f"{pair}-1m-{d}.zip") 
                        for d in daterange(start_date, end_date) 
                        if os.path.exists(os.path.join(download_path, f"{pair}-1m-{d}.zip"))]
    print(f"Filtered {pair} files: {downloaded_files[:5]}, total: {len(downloaded_files)}")
    return downloaded_files

def download_binance_current_day_data(pair, region):
    limit = 1000
    total_minutes = 10080
    requests_needed = (total_minutes + limit - 1) // limit
    dfs = []
    end_time = int(time.time() * 1000)
    
    for i in range(requests_needed):
        start_time = end_time - (limit * 60 * 1000)
        url = f'https://api.binance.{region}/api/v3/klines?symbol={pair}&interval=1m&limit={limit}&endTime={end_time}'
        print(f"Fetching {pair} data batch {i+1}/{requests_needed} from: {url}")
        response = session.get(url)
        response.raise_for_status()
        resp = str(response.content, 'utf-8').rstrip()
        columns = ['start_time', 'open', 'high', 'low', 'close', 'volume', 'end_time', 'volume_usd', 'n_trades', 'taker_volume', 'taker_volume_usd', 'ignore']
        df = pd.DataFrame(json.loads(resp), columns=columns)
        df['date'] = [pd.to_datetime(x+1, unit='ms') for x in df['end_time']]
        df['date'] = df['date'].apply(pd.to_datetime)
        df[["volume", "taker_volume", "open", "high", "low", "close"]] = df[["volume", "taker_volume", "open", "high", "low", "close"]].apply(pd.to_numeric)
        dfs.append(df)
        end_time = int(df['end_time'].iloc[0]) - 1
    
    combined_df = pd.concat(dfs).sort_index()
    print(f"Total {pair} live data rows fetched: {len(combined_df)}")
    return combined_df

def get_coingecko_coin_id(token):
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
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
    print(f"Days: {days}")
    coin_id = get_coingecko_coin_id(token)
    print(f"Coin ID: {coin_id}")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}&api_key={CG_API_KEY}'
    global files
    files = []
    with ThreadPoolExecutor() as executor:
        print(f"Downloading data for {coin_id}")
        name = os.path.basename(url).split("?")[0].replace("/", "_") + ".json"
        executor.submit(download_url, url, download_path, name)
    return files

def download_coingecko_current_day_data(token, CG_API_KEY):
    coin_id = get_coingecko_coin_id(token)
    print(f"Coin ID: {coin_id}")
    url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=1&api_key={CG_API_KEY}'
    response = session.get(url)
    response.raise_for_status()
    resp = str(response.content, 'utf-8').rstrip()
    columns = ['timestamp', 'open', 'high', 'low', 'close']
    df = pd.DataFrame(json.loads(resp), columns=columns)
    df['date'] = [pd.to_datetime(x, unit='ms') for x in df['timestamp']]
    df['date'] = df['date'].apply(pd.to_datetime)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
    return df.sort_index()

if __name__ == "__main__":
    download_binance_daily_data("BTCUSDT", 180, "us", "data/binance")
    download_binance_daily_data("BERAUSDT", 180, "us", "data/binance")
