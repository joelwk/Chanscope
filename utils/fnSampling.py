import pandas as pd
import random
import glob
import os
import logging
from utils.fnProcessing import remove_urls, remove_whitespace

logging.basicConfig(level=logging.INFO)

def stratified_sample(data, strata_column, sample_ratio):
    sample_size = lambda x: max(int(len(x) * sample_ratio), 1)
    return data.groupby(strata_column).apply(lambda x: x.sample(n=sample_size(x)))

def reservoir_sampling(iterator, k):
    reservoir = []
    for i, item in enumerate(iterator):
        if i < k: reservoir.append(item)
        else:
            j = random.randint(0, i)
            if j < k: reservoir[j] = item
    return reservoir

def time_based_sample(data, time_column, freq='H', sample_ratio=None):
    if sample_ratio is None: raise ValueError("Sample_ratio must be provided.")
    data[time_column] = pd.to_datetime(data[time_column])
    if data.index.name != time_column: data.set_index(time_column, inplace=True)
    data = data.resample(freq).apply(lambda x: x.sample(frac=sample_ratio) if len(x) > 0 else x)
    data.reset_index(inplace=True)
    return data

def sample_directory(directory, time_column, freq='H', sample_ratio=None):
    sampled_data = pd.DataFrame()
    total_rows = 0
    for file_name in [f for f in os.listdir(directory) if f.endswith('.parquet')]:
        file_path = os.path.join(directory, file_name)
        data = pd.read_parquet(file_path)
        total_rows += len(data)
        sampled_data = pd.concat([sampled_data, time_based_sample(data, time_column, freq, sample_ratio)])
        logging.info(f'Processed file {file_name}, Total Rows Processed: {total_rows}, Sampled Data Size: {len(sampled_data)}')
    return sampled_data

def get_random_sample_directory(dir_pattern, sample_ratio=None):
    parquet_files = glob.glob(dir_pattern)
    if not parquet_files:
        print("No parquet files found.")
        return None
    data = pd.read_parquet(random.choice(parquet_files))
    if sample_ratio is not None:
        data = data.sample(frac=sample_ratio)
    return data

def reservoir_sample_directory(data, n):
    reservoir = reservoir_sampling(data.thread_id, n)
    return data[data['thread_id'].isin(reservoir)]

def count_total_rows(directory):
    total_rows = 0
    files_count = 0
    for file_name in [f for f in os.listdir(directory) if f.endswith('.parquet')]:
        total_rows += len(pd.read_parquet(os.path.join(directory, file_name)))
        files_count += 1
    print(f'Total number of files processed {files_count} containing: {total_rows} rows')
    return total_rows

def get_sample(directory, sample_ratio):
    data = sample_directory(directory, 'posted_date_time', 'H', sample_ratio) 
    data["text_clean"] = data["posted_comment"].astype(str).apply(remove_whitespace).apply(remove_urls)
    data = data[data['text_clean'] != ''].dropna(subset=['text_clean']).drop_duplicates(subset=['thread_id'])
    data = data.drop_duplicates(subset=['thread_id'])
    # Convert 'thread_id' to numeric, coercing errors to NaN
    data['thread_id'] = pd.to_numeric(data['thread_id'], errors='coerce')
    # Optionally, drop rows with NaN 'thread_id'
    data = data.dropna(subset=['thread_id'])
    # Convert 'thread_id' to integer
    data['thread_id'] = data['thread_id'].astype('int')
    data['YearMonth'] = data['YearMonth'].astype(str)
    return data
