import pandas as pd
import random
import glob
import boto3
import os
import logging
import csv
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
from utils.fnProcessing import remove_urls, remove_whitespace, flip_spam_label, clean_text
import configparser
config_transformer = configparser.ConfigParser()
config_transformer.read('./config.ini')
board_info = config_transformer["board_info"]
params = {key: board_info[key] for key in board_info}

def estimate_max_len(text_data):
    return max(len(s.split()) for s in text_data)

def estimate_vocab_size(text_data):
    unique_words = set(word for s in text_data for word in s.split())
    return len(unique_words)

def iter_sample():
    if initial_sample_size - data.shape[0] < len(file_data):
        file_data = pd.DataFrame(reservoir_sampling(file_data.itertuples(index=False), initial_sample_size - data.shape[0]), columns=file_data.columns)

def load_local_csv(input_dir):
    csv_files = glob.glob(f'{input_dir}/*.csv')
    dataframes = []
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def load_local_parq(input_dir):
    parquet_files = glob.glob(f'{input_dir}/*.parquet')
    if not parquet_files:
        print("No parquet files found in the directory.")
        return pd.DataFrame()
    dataframes = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        for col in df.columns:
            if pd.api.types.is_period_dtype(df[col]):
                df[col] = df[col].astype(str)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    if 'thread_id' in combined_df.columns:
        combined_df['thread_id'] = combined_df['thread_id'].astype(int)
    return combined_df

def save_to_datasets_folder(data, dataset_name):
    data.to_parquet(f'./datasets/{dataset_name}.parquet', index=False)

def save_to_samples_folder(data, dataset_name):
    data.to_parquet(f'./samples/{dataset_name}.parquet', index=False)

def load_and_append_parquets_to_dataset(input_dir, output_dir):
    parquet_files = glob.glob(f'{input_dir}*.parquet')
    dataframes = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dataframes.append(df)
        os.remove(file)
    data = pd.concat(dataframes, ignore_index=True)
    data.loc[:, "text_clean"] = data["posted_comment"].apply(clean_text).astype(str)
    existing_files = os.listdir(output_dir)
    existing_numbers = [int(re.search(r'\d+', file).group()) for file in existing_files if re.search(r'\d+', file)]
    next_file_number = max(existing_numbers) + 1 if existing_numbers else 1
    date_min = pd.to_datetime(data['posted_date_time']).min().strftime('%Y-%m-%d')
    date_max = pd.to_datetime(data['posted_date_time']).max().strftime('%Y-%m-%d')
    total_rows = data.shape[0]
    output_filename = f'dataset_{date_min}_{date_max}_R{total_rows}.parquet'
    data.to_parquet(output_filename, index=False)

def convert_date_columns(df, date_columns):
    def parsing_dates(text):
        for fmt in ('%m-%d-%y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m-%d-%y'):
            try:
                return pd.to_datetime(text, format=fmt)
            except ValueError:
                pass
        raise ValueError('no valid date format found')
    for col in date_columns:
        df[col] = df[col].apply(parsing_dates)
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df
    
def update_file(new_df, existing_df_path, file_type='parquet'):
    new_df = new_df.copy()
    file_exists = os.path.isfile(existing_df_path)
    if not file_exists:
        save_df(new_df, existing_df_path, file_type)
    else:
        df = pd.read_parquet(existing_df_path) 
        df['thread_id'] = df['thread_id'].astype(int)
        new_df['thread_id'] = new_df['thread_id'].astype(int)
        df = pd.concat([df, new_df], ignore_index=True)
        df.drop_duplicates(subset='thread_id', inplace=True)
        for col in df.columns:
            if pd.api.types.is_period_dtype(df[col]):
                df[col] = df[col].astype(str)
        date_columns_to_convert = ['posted_date_time', 'collected_date_time']
        df = convert_date_columns(df, date_columns_to_convert)
        df.to_parquet(existing_df_path)

def load_files(directory_path, num_min_threshold, num_max_threshold, wc_min_threshold, wc_max_threshold):
    dataframes = []
    for file in os.listdir(directory_path):
        if file.endswith(".txt"): 
            number, wc_part = file.split('F')[1].split('WC')
            wc = wc_part.split('_')[0] 
            if (num_min_threshold < int(number) < num_max_threshold) and (wc_min_threshold < int(wc) < wc_max_threshold):
                df = pd.read_table(os.path.join(directory_path, file), sep=',').sort_values(by='date')
                dataframes.append(df)
    return dataframes

def load_and_combine_files(directory_path, num_min_threshold=None, num_max_threshold=None, wc_min_threshold=None, wc_max_threshold=None):
    dataframes = []
    for file in os.listdir(directory_path):
        if file.endswith(".txt"):  
            number, wc_part = file.split('F')[1].split('WC')
            wc = wc_part.split('_')[0]
            if ((num_min_threshold is None or int(number) >= num_min_threshold) and 
                (num_max_threshold is None or int(number) <= num_max_threshold) and 
                (wc_min_threshold is None or int(wc) >= wc_min_threshold) and 
                (wc_max_threshold is None or int(wc) <= wc_max_threshold)):
                df = pd.read_table(os.path.join(directory_path, file), sep=',').sort_values(by='date')
                df['frequency'] = int(number)
                dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df