#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import requests
import pandas as pd
import preprocessing as pp
from datetime import datetime

API_BASE = 'https://api.bitvavo.com/v2/'   # Bitvavo REST base

# Keep same column order you expect elsewhere
LABELS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume'
]

# Map a few common intervals to milliseconds (extend if you need more)
INTERVAL_MS = {
    '1m': 60_000,
    '5m': 5 * 60_000,
    '15m': 15 * 60_000,
    '1h': 60 * 60_000,
    '4h': 4 * 60 * 60_000,
    '1d': 24 * 60 * 60_000,
}

def _normalize_candle_row(row):
    """
    Normalize Bitvavo candle rows (which may be dicts or lists)
    into [open_time, open, high, low, close, volume]
    """
    if isinstance(row, dict):
        # Be forgiving about key casing
        ts = row.get('timestamp') or row.get('Timestamp')
        o = row.get('open') or row.get('Open')
        h = row.get('high') or row.get('High')
        l = row.get('low') or row.get('Low')
        c = row.get('close') or row.get('Close')
        v = row.get('volume') or row.get('Volume')
    else:
        # Expect [Timestamp, Open, High, Low, Close, Volume]
        ts, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]

    return [int(ts), str(o), str(h), str(l), str(c), str(v)]

def get_batch(symbol, interval='1m', start_time=0, retries=5, timeout=30):
    """
    Retrieve a batch of Bitvavo candlesticks for `symbol` (e.g., 'BTC-EUR').
    Bitvavo returns newest->oldest; we reverse to oldest->newest.
    """
    if retries <= 0:
        print('Max retries reached, returning empty dataframe')
        return pd.DataFrame([], columns=LABELS)

    params = {
        'interval': interval,
        'end': start_time
    }

    try:
        response = requests.get(f'{API_BASE}{symbol}/candles', params=params, timeout=timeout)
        response.raise_for_status()

        ratelimit_remaining = response.headers.get('bitvavo-ratelimit-remaining')
        if ratelimit_remaining is not None and int(ratelimit_remaining) < 120:
            if int(ratelimit_remaining) < 60:
                print(f'Low rate limit remaining: {ratelimit_remaining}, sleeping for 2sec..')
                time.sleep(2)
            elif int(ratelimit_remaining) < 30:
                print(f'Low rate limit remaining: {ratelimit_remaining}, sleeping for 5sec..')
                time.sleep(5)
            elif int(ratelimit_remaining) < 10:
                print(f'Low rate limit remaining: {ratelimit_remaining}, sleeping for 60sec..')
                time.sleep(60)
            else:
                time.sleep(1)

        data = response.json()  # typically a list (either lists or dicts)
        if not isinstance(data, list):
            data = []
    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 15 mins...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, retries-1, timeout)
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 15 min...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, retries-1, timeout)
    except ConnectionResetError:
        print('Connection reset by peer, Cooling down for 15 min...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, retries-1, timeout)
    except Exception as e:
        print(f'Unknown error: {e}, Cooling down for 15 min...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, retries-1, timeout)

    if response.status_code == 200:
        # Normalize to your LABELS
        rows = [_normalize_candle_row(r) for r in data]
        df = pd.DataFrame(rows, columns=LABELS)
        if not df.empty:
            # Ensure int64 ms for open_time
            df['open_time'] = df['open_time'].astype(np.int64)
            # strip last minute partial candle if present
            current_minute_open = int(datetime.now().replace(second=0, microsecond=0).timestamp() * 1000)
            df = df[df['open_time'] < current_minute_open]
        return df

    print(f'Got erroneous response back: {response}')
    return pd.DataFrame([], columns=LABELS)

def all_candles_to_csv(base, quote, interval='1m'):
    """
    Collect all candlesticks for base-quote on Bitvavo, write CSV + Parquet.
    """
    market = f'{base}-{quote}'

    oldest_timestamp = int(datetime.now().replace(second=0, microsecond=0).timestamp() * 1000)
    most_recent_file_timestamp = None

    try:
        batches = [pd.read_csv(f'data/{base}-{quote}.csv')]
        most_recent_file_timestamp = batches[-1]['open_time'].max()
    except FileNotFoundError:
        batches = [pd.DataFrame([], columns=LABELS)]
    old_lines = len(batches[-1].index)

    previous_timestamp = None
    while previous_timestamp != oldest_timestamp:
        previous_timestamp = oldest_timestamp

        new_batch = get_batch(
            symbol=market,
            interval=interval,
            start_time=oldest_timestamp
        )

        # If empty assume end of historical data
        if new_batch.empty:
            break

        if not new_batch.empty:
            batches.append(new_batch)

        oldest_timestamp = new_batch['open_time'].min()

        if most_recent_file_timestamp and oldest_timestamp <= most_recent_file_timestamp:
            break

        last_datetime = datetime.fromtimestamp(oldest_timestamp / 1000)
        print(datetime.now(), base, quote, interval, str(last_datetime) + (20 * ' '), end='\r', flush=True)

    # Write Parquet (cleaned)
    parquet_name = f'{base}-{quote}.parquet'
    full_path = f'compressed/{parquet_name}'
    df = pd.concat(batches, ignore_index=True)
    df = pp.quick_clean(df)

    pp.write_raw_to_parquet(df, full_path)

    # Write CSV if new data gathered
    if len(batches) > 1:
        df.to_csv(f'data/{base}-{quote}.csv', index=False)
        return len(df.index) - old_lines
    return 0

def main():
    """
    Main loop; iterate all Bitvavo markets (EUR/USDC quotes by default) and update datasets.
    """
    # Discover all markets
    markets_resp = requests.get(f'{API_BASE}markets', timeout=30)
    all_symbols = pd.DataFrame(markets_resp.json())  # expect fields: market, base, quote, ...
    # Filter desired quotes
    all_symbols = all_symbols[all_symbols['quote'].isin(['EUR', 'USDC'])]

    # Optional: blacklist bases if you want
    blacklist = []
    for coin in blacklist:
        all_symbols = all_symbols[all_symbols['base'] != coin]

    # Build (base, quote) tuples
    filtered_pairs = [tuple(x) for x in all_symbols[['base', 'quote']].to_records(index=False)]

    # debug, only one pair
    # filtered_pairs = [('PRIME', 'EUR')]

    # Ensure folders
    os.makedirs('data', exist_ok=True)
    os.makedirs('compressed', exist_ok=True)

    # Full update on all pairs
    n_count = len(filtered_pairs)
    for n, pair in enumerate(filtered_pairs, 1):
        try:
            base, quote = pair
            new_lines = all_candles_to_csv(base=base, quote=quote)
            if new_lines > 0:
                print(f'{datetime.now()} {n}/{n_count} Wrote {new_lines} new lines to file for {base}-{quote}')
            else:
                print(f'{datetime.now()} {n}/{n_count} Already up to date with {base}-{quote}')
        except Exception as e:
            print(f'Error with {pair}: {e}')
            continue

if __name__ == '__main__':
    main()
