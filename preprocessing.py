import os
from datetime import date, datetime, timedelta
import pandas as pd


def set_dtypes(df):
    """
    set datetimeindex and convert all columns in pd.df to their proper dtype
    assumes csv is read raw without modifications; pd.read_csv(csv_filename)"""

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time', drop=True)

    df = df.astype(dtype={
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'quote_asset_volume': 'float64',
    })

    return df


def set_dtypes_compressed(df):
    """Create a `DatetimeIndex` and convert all critical columns in pd.df to a dtype with low
    memory profile. Assumes csv is read raw without modifications; `pd.read_csv(csv_filename)`."""

    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time', drop=True)

    df = df.astype(dtype={
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'float32',
        'quote_asset_volume': 'float32',
    })

    return df


def assert_integrity(df):
    """make sure no rows have empty cells or duplicate timestamps exist"""

    assert df.isna().all(axis=1).any() == False
    assert df['open_time'].duplicated().any() == False


def quick_clean(df):
    """clean a raw dataframe"""

    # drop dupes
    dupes = df['open_time'].duplicated().sum()
    if dupes > 0:
        df = df[df['open_time'].duplicated() == False]

    # sort by timestamp, oldest first
    df.sort_values(by=['open_time'], ascending=False)

    # just a doublcheck
    assert_integrity(df)

    return df


def add_missing_minutes(initial_data):
    """
    Repeat entries to fill up for missing minutes.

    """
    clean_data = []
    previous_row = []
    for row in initial_data:
        if len(previous_row) and row['datetime'] - previous_row['datetime'] > timedelta(minutes=1):
            current = previous_row.copy()
            while current['datetime'] + timedelta(minutes=1) < row['datetime']:
                current['datetime'] = current['datetime'] + timedelta(minutes=1)
                current['volume'] = 0
                current['quote_asset_volume'] = 0
                current['low'] = current['close']
                current['high'] = current['close']
                current['open'] = current['close']
                missing_row = current.copy()
                clean_data.append(missing_row)

        previous_row = row.copy()
        clean_data.append(row)

    return clean_data

def add_missing_minutes_df(df):
    """
    Repeat entries of the given dataframe to fill up for missing minutes.
    
    """
    historical_data_dict = add_missing_minutes(df.to_dict('records'))
    return pd.DataFrame(historical_data_dict)


def write_raw_to_parquet(df, full_path):
    """takes raw df and writes a parquet to disk"""

    df_copy = df.copy()
    df_copy = set_dtypes_compressed(df_copy)

    # give all pairs the same nice cut-off
    #df_copy = df_copy[df_copy.index < str(date.today())]

    # post-processing for FDA
    df_copy['datetime'] = df_copy.index
    df_copy['datetime'] = df_copy['datetime'].dt.floor('Min')
    df_copy = df_copy.drop_duplicates(subset=['datetime'], keep='first')
    df_copy.reset_index(drop=True, inplace=True)
    df_copy = add_missing_minutes_df(df_copy)
    df_copy.reset_index(drop=True, inplace=True)

    df_copy.to_parquet(full_path)


def groom_data(dirname='data'):
    """go through data folder and perform a quick clean on all csv files"""

    for filename in os.listdir(dirname):
        if filename.endswith('.csv'):
            full_path = f'{dirname}/{filename}'
            quick_clean(pd.read_csv(full_path)).to_csv(full_path)


def compress_data(dirname='data'):
    """go through data folder and rewrite csv files to parquets"""

    os.makedirs('compressed', exist_ok=True)
    for filename in os.listdir(dirname):
        if filename.endswith('.csv'):
            full_path = f'{dirname}/{filename}'

            df = pd.read_csv(full_path)

            new_filename = filename.replace('.csv', '.parquet')
            new_full_path = f'compressed/{new_filename}'
            write_raw_to_parquet(df, new_full_path)
