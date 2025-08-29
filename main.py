#!/usr/bin/env python
# coding: utf-8

import json
import os
import random
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
import numpy as np
import requests
import pandas as pd
import preprocessing as pp

START_TIME = (datetime.now() - timedelta(hours=0, minutes=5))
API_BASE = 'https://api.bitvavo.com/v2/'   # Bitvavo REST base

# Keep same column order you expect elsewhere
LABELS = [
    'open_time',
    'open',
    'high',
    'low',
    'close',
    'volume',
    'quote_asset_volume'
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
    into [open_time, open, high, low, close, volume, quote_asset_volume]
    with ms timestamps and string numeric fields preserved (like your flow).
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
        # Some clients return strings, others numbers; keep as strings like Bitvavo flow
        ts, o, h, l, c, v = row[0], row[1], row[2], row[3], row[4], row[5]

    return [int(ts), str(o), str(h), str(l), str(c), str(v), np.nan]  # Bitvavo doesn’t return quote volume

def get_batch(symbol, interval='1m', start_time=0, limit=1440, retries=5, timeout=30):
    """
    Retrieve a batch of Bitvavo candlesticks for `symbol` (e.g., 'BTC-EUR').
    Bitvavo returns newest->oldest; we reverse to oldest->newest.
    We bound the request with [start, end] to avoid fetching the entire history.
    """
    if retries <= 0:
        print('Max retries reached, returning empty dataframe')
        return pd.DataFrame([], columns=LABELS)

    # Determine an end bound roughly covering `limit` candles from start_time
    interval_ms = INTERVAL_MS.get(interval)
    if not interval_ms:
        # Fallback assume minutes if last char is 'm'
        if interval.endswith('m') and interval[:-1].isdigit():
            interval_ms = int(interval[:-1]) * 60_000
        elif interval.endswith('h') and interval[:-1].isdigit():
            interval_ms = int(interval[:-1]) * 60 * 60_000
        elif interval.endswith('d') and interval[:-1].isdigit():
            interval_ms = int(interval[:-1]) * 24 * 60 * 60_000
        else:
            # Default to 1m semantics
            interval_ms = 60_000

    # Use a conservative window to avoid overshooting too far
    end_time = start_time + limit * interval_ms - 1

    params = {
        'interval': interval,
        'start': start_time,
        'end': end_time,
        'limit': limit
    }

    try:
        response = requests.get(f'{API_BASE}{symbol}/candles', params=params, timeout=timeout)
        response.raise_for_status()

        ratelimit_remaining = response.headers.get('bitvavo-ratelimit-remaining')
        if ratelimit_remaining is not None and int(ratelimit_remaining) < 120:
            if int(ratelimit_remaining) < 60:
                print(f'Low rate limit remaining: {ratelimit_remaining}')
                print(f'Sleeping for 2 seconds..')
                time.sleep(2)
            elif int(ratelimit_remaining) < 10:
                print(f'Low rate limit remaining: {ratelimit_remaining}')
                print(f'Sleeping for 60 seconds..')
                time.sleep(60)
            else:
                time.sleep(1)

        data = response.json()  # typically a list (either lists or dicts)
        if not isinstance(data, list):
            data = []
        # Bitvavo returns newest->oldest; make ascending for downstream logic
        data.reverse()
    except requests.exceptions.ConnectionError:
        print('Connection error, Cooling down for 15 mins...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, limit, retries-1, timeout)
    except requests.exceptions.Timeout:
        print('Timeout, Cooling down for 15 min...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, limit, retries-1, timeout)
    except ConnectionResetError:
        print('Connection reset by peer, Cooling down for 15 min...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, limit, retries-1, timeout)
    except Exception as e:
        print(f'Unknown error: {e}, Cooling down for 15 min...')
        time.sleep(15 * 60)
        return get_batch(symbol, interval, start_time, limit, retries-1, timeout)

    if response.status_code == 200:
        # Normalize to your LABELS
        rows = [_normalize_candle_row(r) for r in data]
        df = pd.DataFrame(rows, columns=LABELS)
        # Ensure int64 ms for open_time
        if not df.empty:
            df['open_time'] = df['open_time'].astype(np.int64)
            df = df[df.open_time < START_TIME.timestamp() * 1000]
        return df

    print(f'Got erroneous response back: {response}')
    return pd.DataFrame([], columns=LABELS)

def all_candles_to_csv(base, quote, interval='1m'):
    """
    Collect all candlesticks for base-quote on Bitvavo, write CSV + Parquet.
    """
    market = f'{base}-{quote}'

    try:
        batches = [pd.read_csv(f'data/{base}-{quote}.csv')]
        last_timestamp = batches[-1]['open_time'].max()
        new_file = False
    except FileNotFoundError:
        batches = [pd.DataFrame([], columns=LABELS)]
        start_datetime_string = "2019-04-01 00:00:00"
        last_timestamp = int(datetime.strptime(start_datetime_string, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
        new_file = True
    old_lines = len(batches[-1].index)

    previous_timestamp = None
    while previous_timestamp != last_timestamp:
        previous_timestamp = last_timestamp

        new_batch = get_batch(
            symbol=market,
            interval=interval,
            start_time=last_timestamp + 1,
            limit=1440
        )

        # If empty and not a new file, assume up to date
        if new_batch.empty and not new_file:
            break

        if new_batch.empty:
            # advance a day to skip gaps (no-trade periods)
            last_timestamp = last_timestamp + 86_400_000
        else:
            # Bitvavo doesn't guarantee contiguous candles; jump by the batch span
            last_timestamp = new_batch['open_time'].max() + (300 * INTERVAL_MS.get(interval, 60_000))

        if not new_batch.empty:
            batches.append(new_batch)
            new_file = False

        last_datetime = datetime.fromtimestamp(last_timestamp / 1000)
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

    # # debug only btc
    # all_symbols = all_symbols[all_symbols['base'].isin(['BTC'])]

    # Optional: blacklist bases if you want
    blacklist = []
    for coin in blacklist:
        all_symbols = all_symbols[all_symbols['base'] != coin]

    # Build (base, quote) tuples
    filtered_pairs = [tuple(x) for x in all_symbols[['base', 'quote']].to_records(index=False)]

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
