import pandas as pd
import random
import boto3
import os
import csv
from scipy.stats import ks_2samp
import pandas as pd
import random
import boto3
import os
from matplotlib import pyplot as plt

def stratified_sample(data, strata_column, sample_ratio):
    """
    Perform stratified sampling on a DataFrame.
    
    Parameters:
    - data: DataFrame to sample from.
    - strata_column: Column to use for stratification.
    - sample_ratio: Ratio of each stratum to sample.
    
    Returns:
    - DataFrame of the sampled data.
    """
    sample_size = lambda x: max(int(len(x) * sample_ratio), 1)
    return data.groupby(strata_column).apply(lambda x: x.sample(n=sample_size(x)))

def time_based_sample(data, time_column, freq='H'):
    """
    Perform time-based sampling on a DataFrame.
    
    Parameters:
    - data: DataFrame to sample from.
    - time_column: Column with datetime data to use for sampling.
    - freq: Frequency to sample at, defaults to 'H' for hourly.
    
    Returns:
    - DataFrame of the sampled data.
    """
    data[time_column] = pd.to_datetime(data[time_column])
    data = data.set_index(time_column)
    data = data.resample(freq).apply(lambda x: x.sample(n=min(len(x), 1)) if len(x) > 0 else None)
    data = data.dropna(how='all')  # Drop rows that are all NaN
    data = data.reset_index()
    return data

def reservoir_sampling(iterator, k):
    """
    Reservoir sampling algorithm to get a sample of size k from an iterator.
    Parameters:
    - iterator: Iterator to sample from.
    - k: Number of samples to draw.
    Returns:
    - List of k samples.
    """
    reservoir = []
    for i, item in enumerate(iterator):
        if i < k:
            # For the first k items, just append them to the reservoir
            reservoir.append(item)
        else:
            # For the i-th item (where i >= k), replace a random item in the reservoir with it with probability k/i
            j = random.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir

def get_initial_stratified_sample(source='local', source_directory=None, bucket=None, s3_prefix=None, initial_sample_ratio=1.0, 
                                  strata_column=None, time_column=None, freq=None, file_contains=None, 
                                  file_type='csv', file_ratio=1.0):
    """
    Get an initial stratified sample from local files or S3.
    Parameters:
    - source: Source of the data ('local' or 's3').
    - source_directory: Directory of the data if source is local.
    - bucket: S3 bucket name if source is s3.
    - s3_prefix: S3 prefix if source is s3.
    - initial_sample_ratio: Initial ratio of data to sample.
    - strata_column: Column to use for stratification.
    - time_column: Column with datetime data to use for sampling.
    - freq: Frequency to sample at, defaults to 'H' for hourly.
    - file_contains: String to filter files by.
    - file_type: Type of file ('csv' or 'parquet').
    - file_ratio: Ratio of files to consider for sampling.
    Returns:
    - DataFrame of the sampled data.
    """
    if file_type not in ['csv', 'parquet']:
        print("Invalid file type. Choose 'csv' or 'parquet'.")
        return pd.DataFrame()
    if source == 'local':
        # Get a list of all files of the specified type in the directory
        files = [f for f in os.listdir(source_directory) 
                 if f.endswith(f'.{file_type}') and file_contains in f]
    else:  # 's3'
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        files = [item['Key'] for item in response['Contents'] if item['Key'].endswith(f'.{file_type}')]
    if not files:
        print(f"No {file_type} files found.") 
        return pd.DataFrame()
    # Select a subset of files according to the file_ratio
    selected_files = random.sample(files, int(len(files) * file_ratio))
    data = pd.DataFrame()
    for file_key in selected_files:
        if source == 'local':
            temp_file = os.path.join(source_directory, file_key)
        else:  # 's3'
            temp_file = "temp.csv"
            s3.download_file(bucket, file_key, temp_file)
        if file_type == 'csv':
            file_data = pd.read_csv(temp_file, index_col=False)
        else:  # 'parquet'
            file_data = pd.read_parquet(temp_file)
        # Sample initial_sample_ratio of rows from the selected file
        sample_size = int(initial_sample_ratio * len(file_data))
        file_data = pd.DataFrame(reservoir_sampling(file_data.itertuples(index=False), sample_size), columns=file_data.columns)
        # Apply stratified sampling if strata_column exists in data
        if strata_column is not None and strata_column in file_data.columns:
            file_data = stratified_sample(file_data, strata_column, initial_sample_ratio)    
        # Apply time-based sampling
        if time_column is not None and freq is not None:
            file_data = time_based_sample(file_data, time_column, freq)
        data = pd.concat([data, file_data])
        if source == 's3':
            os.remove(temp_file)
    unnamed_cols = data.filter(regex='^Unnamed').columns
    data = data.drop(columns=unnamed_cols)
    return data

def get_sufficient_stratified_sample(source='local', source_directory=None, bucket=None, s3_prefix=None, 
                                     sample_ratio=None, strata_column=None, time_column=None, freq=None, 
                                     file_keys=None, file_ratio=1.0, quality_check_column=None, file_contains=None,
                                     reservoir_size=10000, min_sample_size=1000, file_type='csv'):
    # Check file_type
    if file_type not in ['csv', 'parquet']:
        print("Invalid file type. Choose 'csv' or 'parquet'.")
        return pd.DataFrame()
    if source == 'local':
        file_objects = [f for f in os.listdir(source_directory) 
                        if f.endswith(f'.{file_type}') and file_contains in f and
                        (file_keys is None or any(key in f for key in file_keys))]
    else:  # 's3'
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        file_objects = [item['Key'] for item in response['Contents'] 
                        if item['Key'].endswith(f'.{file_type}') and 
                        (file_keys is None or any(key in item['Key'] for key in file_keys))]
    if not file_objects:
        print(f"No {file_type} files found.")
        return pd.DataFrame()
    # Select a subset of files according to the file_ratio
    selected_files = random.sample(file_objects, int(len(file_objects) * file_ratio))
    data = pd.DataFrame()
    for file_key in selected_files:
        if source == 'local':
            temp_file = os.path.join(source_directory, file_key)
        else:  # 's3'
            temp_file = "temp." + file_type
            s3.download_file(bucket, file_key, temp_file)
        if file_type == 'csv':
            file_data = pd.read_csv(temp_file, index_col=False) 
        elif file_type == 'parquet':
            file_data = pd.read_parquet(temp_file)
        # Apply stratified sampling if strata_column exists in data
        if strata_column is not None and strata_column in file_data.columns:
            file_data = stratified_sample(file_data, strata_column, sample_ratio)
        # Apply time-based sampling
        if time_column is not None and freq is not None:
            file_data = time_based_sample(file_data, time_column, freq)
        # Apply random sampling
        if sample_ratio is not None:
            file_data = file_data.sample(frac=sample_ratio)
        data = pd.concat([data, file_data])
        if source == 's3':
            os.remove(temp_file)
        unnamed_cols = data.filter(regex='^Unnamed').columns
        data = data.drop(columns=unnamed_cols)
        # Quality check: compare the distribution of the quality_check_column in the sample and the population
        if quality_check_column is not None and quality_check_column in data.columns:
            population_sample = []
            for file_key in file_objects:
                if source == 'local':
                    temp_file = os.path.join(source_directory, file_key)
                else:  # 's3'
                    temp_file = "temp." + file_type
                    s3.download_file(bucket, file_key, temp_file)
                if file_type == 'csv':
                    with open(temp_file, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        population_sample.extend(reservoir_sampling((row for row in reader), reservoir_size))
                elif file_type == 'parquet':
                    parquet_file = pq.ParquetFile(temp_file)
                    for i in range(parquet_file.num_row_groups):
                        file_data = parquet_file.read_row_group(i).to_pandas()
                        population_sample.extend(reservoir_sampling(file_data.itertuples(index=False), reservoir_size))
                if source == 's3':
                    os.remove(temp_file)
            population_freqs = population_sample[quality_check_column].value_counts()
            sample_freqs = data[quality_check_column].value_counts()

            # Ensure the two frequency series have the same indices
            all_categories = np.union1d(population_freqs.index, sample_freqs.index)
            population_freqs = population_freqs.reindex(all_categories, fill_value=0)
            sample_freqs = sample_freqs.reindex(all_categories, fill_value=0)

            # Now we compute the chi-square test statistic
            chi2, pvalue, _, _ = chi2_contingency([population_freqs, sample_freqs])
            if pvalue < 0.05:
                print(f"Adjusting the sampling strategy...") 
    # Do not reset the data DataFrame if the sample is not representative
    return data

def get_simple_sample(source='local', source_directory=None, bucket=None, s3_prefix=None, 
                      initial_sample_size=10000, file_keys=None, file_type='parquet', file_ratio=1.0):
    if source not in ['local', 's3']:
        print("Invalid source. Choose 'local' or 's3'.")
        return pd.DataFrame()
    if file_type not in ['csv', 'parquet']:
        print("Invalid file type. Choose 'csv' or 'parquet'.")
        return pd.DataFrame()
    if source == 'local':
        files = [f for f in os.listdir(source_directory) 
                 if f.endswith(f'.{file_type}') and 
                 (file_keys is None or f[:-len(file_type)-1] in file_keys)]
    else:  # 's3'
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
        files = [item['Key'] for item in response['Contents'] if item['Key'].endswith(f'.{file_type}')]
    if not files:
        print(f"No {file_type} files found.")
        return pd.DataFrame()
    # Select a subset of files according to the file_ratio
    selected_files = random.sample(files, int(len(files) * file_ratio))
    data = pd.DataFrame()
    for file_key in selected_files:
        if source == 'local':
            temp_file = os.path.join(source_directory, file_key)
        else:  # 's3'
            temp_file = "temp." + file_type
            s3.download_file(bucket, file_key, temp_file)
        if file_type == 'csv':
            chunksize = max(1, initial_sample_size - data.shape[0])
            chunks = pd.read_csv(temp_file, chunksize=chunksize)
            for file_data in chunks:
                if data.shape[0] >= initial_sample_size:
                    break
                if initial_sample_size - data.shape[0] < len(file_data):
                    file_data = pd.DataFrame(reservoir_sampling(file_data.itertuples(index=False), initial_sample_size - data.shape[0]), columns=file_data.columns)
                data = pd.concat([data, file_data])
        else:  # 'parquet'
            parquet_file = pq.ParquetFile(temp_file)
            for i in range(parquet_file.num_row_groups):
                file_data = parquet_file.read_row_group(i).to_pandas()

                if data.shape[0] >= initial_sample_size:
                    break
                if initial_sample_size - data.shape[0] < len(file_data):
                    file_data = pd.DataFrame(reservoir_sampling(file_data.itertuples(index=False), initial_sample_size - data.shape[0]), columns=file_data.columns)
                data = pd.concat([data, file_data])
        if source == 's3':
            os.remove(temp_file)
    return data

def get_random_sample_parquet_from_local(dir_pattern, sample_ratio=None):
    parquet_files = glob.glob(dir_pattern)
    if not parquet_files:
        print("No parquet files found.")
        return None
    # Select a random file
    random_file_path = random.choice(parquet_files)
    data = pd.read_parquet(random_file_path)
    
    if sample_ratio is not None:
        data = data.sample(frac=sample_ratio)
    return data

def reservoir_sample_dataframe(data, n):
    """
    Returns @param n random items from @param iterable.
    """
    reservoir = []
    iterable = data.thread_id
    for t, item in enumerate(iterable):
        if t < n:
            reservoir.append(item)
        else:
            m = random.randint(0,t)
            if m < n:
                reservoir[m] = item
    data = data[data['thread_id'].isin(reservoir)]
    return data