import numpy as np
import pandas as pd
import os


def split_raw_data(import_folder_path, export_folder_path, chunk_length=3000):
    """
    Import all csv files in the folder and split them into one minute chunks.
    """
    csv_files = [f for f in os.listdir(import_folder_path) if 'csv' in f]
    export_coloumns = ['timestamp', 'accel.x', 'accel.y', 'accel.z']

    for f in csv_files:
        df = pd.read_csv(import_folder_path + f)
        # Check if the data contains needed columns
        if all(col in df.columns for col in export_coloumns):
            print(f'{f} contains needed columns. Processing...')
            # Drop the first 10 seconds and last 10 seconds
            df = df.iloc[500:-500]
            # Split the data into 1 minute chunks
            nrows = len(df)
            nchunks = nrows // chunk_length
            space_btw_chunk = (nrows % chunk_length) // nchunks
            for i in range(nchunks):
                start_index = i * (chunk_length + space_btw_chunk)
                end_index = start_index + chunk_length
                chunk = df.iloc[start_index:end_index][export_coloumns]
                # Export data
                time_stamp = pd.to_datetime(chunk['timestamp'].iloc[0], unit='ms', utc=True).tz_convert('Europe/Paris')
                time_stamp = time_stamp.strftime('%Y-%m-%d %H:%M:%S%z')
                file_name = f.split('-')[0] + time_stamp
                chunk.to_csv(export_folder_path + f'{file_name}.csv', index=False)
                print(f'Data chunk {i} exported as {file_name}.csv')
        else:
            print(f'{f} does not contain needed columns. Skipping...')


def merge_raw_data(import_folder_path, keyword='handflapping'):
    """
    Import all csv files in the folder and split them into one minute chunks.
    """
    csv_files = [f for f in os.listdir(import_folder_path) if 'csv' in f and keyword in f]
    export_coloumns = ['timestamp', 'accel.x', 'accel.y', 'accel.z']
    merged_df = pd.DataFrame()

    for f in csv_files:
        df = pd.read_csv(import_folder_path + f)
        # Check if the data contains needed columns
        if all(col in df.columns for col in export_coloumns):
            print(f'{f} contains needed columns. Processing...')
            # Drop the first 10 seconds and last 10 seconds
            df = df.iloc[500:-500][export_coloumns]
            # Split the data into 1 minute chunks
            merged_df = pd.concat([merged_df, df], ignore_index=False)
        else:
            print(f'{f} does not contain needed columns. Skipping...')

    merged_df.reset_index(drop=True, inplace=True)

    return merged_df


def add_timestamp_motion_data(df):
    df['timestamp_utc_ms'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp_paris_ms'] = df['timestamp_utc_ms'].dt.tz_convert('Europe/Paris')
    df['timestamp_paris_s'] = df['timestamp_paris_ms'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    df['timestamp_paris_s'] = pd.to_datetime(df['timestamp_paris_s'])
    return df


def add_timestamp_annotation_data(df):
    # Combine date and timestamp into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['timestamp'])
    # Set timezone to Paris
    df['datetime'] = df['datetime'].dt.tz_localize('Europe/Paris')
    return df


def convert_boolean(col):
    return col.where(col.isna(), col.astype(bool))


def acceleration_standardization(df):
    df['accel.magnitude'] = np.sqrt(
        df['accel.x'] ** 2 + df['accel.y'] ** 2 + df['accel.z'] ** 2
        )

    df['accel.x.standardized'] = df['accel.x'] / df['accel.magnitude']
    df['accel.y.standardized'] = df['accel.y'] / df['accel.magnitude']
    df['accel.z.standardized'] = df['accel.z'] / df['accel.magnitude']

    return df
