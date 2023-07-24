import html
import os
import glob
import random
import gzip
import re
import boto3
import pickle
import io
from unicodedata import normalize
import html
import configparser
import pandas as pd
import string
import numpy as np
import tensorflow as tf

config_transformer = configparser.ConfigParser()
config_transformer.read('./.config.ini')
board_info = config_transformer["board_info"]
source_data_s3 = config_transformer["source_data_s3"]
s3_bucket_name = source_data_s3['s3_bucket']
s3_bucket_data = source_data_s3['s3_bucket_data']
s3_bucket_processed = source_data_s3['s3_bucket_processed']
processed_data_gz = source_data_s3['processed_data_gz']
s3_bucket_batchprocessed = source_data_s3['s3_bucket_batchprocessed']

def iter_sample():
    if initial_sample_size - data.shape[0] < len(file_data):
        file_data = pd.DataFrame(reservoir_sampling(file_data.itertuples(index=False), initial_sample_size - data.shape[0]), columns=file_data.columns)

def load_local_csv(input_dir):
    # Get a list of all CSV files in the input directory
    csv_files = glob.glob(f'{input_dir}/*.csv')
    # Initialize an empty list to store dataframes
    dataframes = []
    # Loop through all CSV files and append into one dataframe
    for file in csv_files:
        df = pd.read_csv(file)
        dataframes.append(df)
    # Concatenate all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def load_local_parq(input_dir):
    # Get a list of all Parquet files in the input directory
    parquet_files = glob.glob(f'{input_dir}/*.parquet')
    # Initialize an empty list to store dataframes
    dataframes = []
    # Loop through all Parquet files and append into one dataframe
    for file in parquet_files:
        df = pd.read_parquet(file)
        dataframes.append(df)
    # Concatenate all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def save_to_datasets_folder(data, dataset_name):
    # Save the dataframe to a Parquet file
    data.to_parquet(f'./.datasets/{dataset_name}.parquet', index=False)

def save_to_samples_folder(data, dataset_name):
    # Save the dataframe to a Parquet file
    data.to_parquet(f'./.samples/{dataset_name}.parquet', index=False)

# Load all samples and delete them afterwards
def load_and_append_parquets_to_dataset(input_dir, output_dir):
    # Get a list of all Parquet files in the input directory
    parquet_files = glob.glob(f'{input_dir}*.parquet')
    # Initialize an empty list to store dataframes
    dataframes = []
    # Loop through all Parquet files and append into one dataframe
    for file in parquet_files:
        df = pd.read_parquet(file)
        dataframes.append(df)
        # Delete the file after it has been read and appended
        os.remove(file)
    # Concatenate all dataframes in the list
    data = pd.concat(dataframes, ignore_index=True)
    # Determine the next file number
    existing_files = os.listdir(output_dir)
    existing_numbers = [int(re.search(r'\d+', file).group()) for file in existing_files if re.search(r'\d+', file)]
    next_file_number = max(existing_numbers) + 1 if existing_numbers else 1
    # Determine min and max date
    date_min = pd.to_datetime(data['posted_date_time']).min().strftime('%Y-%m-%d_%H-%M-%S')
    date_max = pd.to_datetime(data['posted_date_time']).max().strftime('%Y-%m-%d_%H-%M-%S')
    # Determine total rows
    total_rows = data.shape[0]
    # Construct the output filename
    output_filename = f'{output_dir}/dataset_{next_file_number}_{date_min}_{date_max}_R{total_rows}.parquet'
    # Save the dataframe to a new Parquet file in the output directory
    data.to_parquet(output_filename, index=False)

def update_file(new_df, existing_df_path, file_type='csv'):
    # Check if file exists
  file_exists = os.path.isfile(existing_df_path)
  if not file_exists:
    # File doesn't exist, just save new DF
    save_df(new_df, existing_df_path, file_type)
  # Read in existing DF
  df = pd.read_csv(existing_df_path) 
  # Ensure columns are integers
  df['thread_id'] = df['thread_id'].astype(int)
  new_df['thread_id'] = new_df['thread_id'].astype(int)
  # Concatenate new DataFrame  
  df = pd.concat([df, new_df], ignore_index=True)
  # Deduplicate 
  df.drop_duplicates(subset='thread_id', inplace=True)
  # Write updated DataFrame
  df.to_csv(existing_df_path, index=False)

def read_df(path, file_type):
  if file_type == 'csv':
    return pd.read_csv(path)
  elif file_type == 'parquet':
    return pd.read_parquet(path)
  else:
    raise ValueError("Invalid file type")
    
def save_df(df, path, file_type):
  if file_type == 'csv':
    df.to_csv(path, index=False)
  elif file_type == 'parquet':
    df.to_parquet(path, index=False)
  else:
    raise ValueError("Invalid file type")

def load_files(directory_path, num_min_threshold, num_max_threshold, wc_min_threshold, wc_max_threshold):
    dataframes = []
    for file in os.listdir(directory_path):
        if file.endswith(".txt"):  # ensures we're working with .txt files
            # split file name and get the number and wc part
            number, wc_part = file.split('F')[1].split('WC')
            wc = wc_part.split('_')[0]  # remove trailing parts after 'WC'
            if (num_min_threshold < int(number) < num_max_threshold) and (wc_min_threshold < int(wc) < wc_max_threshold):
                df = pd.read_table(os.path.join(directory_path, file), sep=',').sort_values(by='date')
                dataframes.append(df)
    return dataframes

def load_and_combine_files(directory_path, num_min_threshold=None, num_max_threshold=None, wc_min_threshold=None, wc_max_threshold=None):
    dataframes = []
    for file in os.listdir(directory_path):
        if file.endswith(".txt"):  # ensures we're working with .txt files
            # split file name and get the number and wc part
            number, wc_part = file.split('F')[1].split('WC')
            wc = wc_part.split('_')[0]  # remove trailing parts after 'WC'
            if ((num_min_threshold is None or int(number) >= num_min_threshold) and 
                (num_max_threshold is None or int(number) <= num_max_threshold) and 
                (wc_min_threshold is None or int(wc) >= wc_min_threshold) and 
                (wc_max_threshold is None or int(wc) <= wc_max_threshold)):
                df = pd.read_table(os.path.join(directory_path, file), sep=',').sort_values(by='date')
                df['frequency'] = int(number)  # adding frequency column
                dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Defining a function that checks a sample of the data and generates a histogram and KDE plot
def check_sample(data):
    # Displays the first 2 rows of the DataFrame
    data.head(2)
    # Plots a histogram and KDE of the data
    plot_hist(data)
    plot_kde(data)