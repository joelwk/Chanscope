# Import the necessary libraries
import requests
import boto3
import json
import csv
import configparser
import logging
import datetime
import pandas as pd

# Set up the Amazon S3 resource and client
s3 = boto3.resource('s3')
s3_c = boto3.client('s3')

# Read and parse the configuration file
config_obj = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
config_obj.read('configfile_gather.ini')
chanscope_param = config_obj["board_info"]
s3_params = config_obj["s3_event"]

# Parse strings from the configuration
url = chanscope_param['url']
threads = chanscope_param['threads']
thread_number = chanscope_param['thread_number']
thread_cmt_number = chanscope_param['thread_cmt_number']
collected_dt = chanscope_param['collected_dt']
posted_dt = chanscope_param['posted_dt']
date_ = chanscope_param['date_']
now_ = chanscope_param['now_']
time_ = chanscope_param['time_']
omit_ids = chanscope_param['omit_ids']
# Parse integers from the configuration
omit_ids = list(map(int, config_obj['board_info'].getlist('omit_ids')))
# Parse list from the configuration
boards = config_obj['board_info'].getlist('boards')

bucket_name = s3_params['s3_bucket']
path_temp = s3_params['path_temp']
path_padding = s3_params['path_padding']
path_Ftype = s3_params['path_Ftype']

# Function to handle AWS Lambda execution
def lambda_handler(event, context):
    # Iterate over all boards
    for _board_ in boards:
        # Send a GET request to the board URL
        response = requests.get(f'{url}/{_board_}/catalog.json')
        # Parse the JSON response
        response_json_threads = response.json()
        thread_no = []
        # Parse thread numbers
        for post_item in response_json_threads:
            for line in post_item[threads]:
                for k,v in line.items():
                    if k == thread_number:
                        thread_no.append(v)

        # Retrieve each comment from each thread
        id_count = len(thread_no)
        x = 0
        data_list = []
        if x != id_count:
            for item in thread_no:
                response_json_items = requests.get(f'{url}/{_board_}/thread/{item}.json')
                data_json = response_json_items.json()
                data_list.append(data_json)
                x += 1

        # Extract all comments from each thread
        data_all = []
        for item in data_list:
            for line in item[thread_cmt_number]:
                data_all.append(line)
        data = pd.DataFrame(data_all)

        # Function to get the current date and time
        def getPulled_dt():
            dt_now = datetime.datetime.now()
            current_date = (
                dt_now.strftime('%Y-%m-%d %H:%M:%S'))
            return current_date

        # Get the current date and time and add it as a new column
        pulled_dt = getPulled_dt()
        data[collected_dt] = pd.to_datetime(pulled_dt).floor('min')

        # Create separate date/time columns
        data[[date_, now_]] = data[now_].str.split('(', expand=True)
        data[[now_, time_]] = data[now_].str.split(')', expand=True)
        data[posted_dt] = pd.to_datetime(data[date_] + data[time_], format='%m/%d/%y%H:%M:%S')
        data[time_] = pd.to_datetime(data[time_], errors='ignore', utc=True).dt.floor('min')
        data[time_]  = data[time_].apply(lambda x: x.strftime("%H:%M:%S"))

        # Drop catalog threads
        ids = omit_ids
        # Drop rows from the above list
        data = data[data.no.isin(ids) == False]

        # Define the save path and save the DataFrame as a CSV file
        save_path = f'{path_temp}/{_board_}_{path_padding}_{pulled_dt}{path_Ftype}'
        data.to_csv(save_path, index=False)

        # Open the file and upload it to the S3 bucket
        with open(save_path, "rb") as f:
            s3_c.upload_fileobj(f, bucket_name, save_path)
    
    # Return a dictionary with the status of the response
    return {
        'status': print(response)
    }