import os
import pandas as pd
from collections import Counter
import json
import os
import pandas as pd
from collections import Counter

def process_data(df, column):
    # Step 1: Dynamic frequency logging
    dynamic_frequency_logging(df, column)
    # Specify directories
    directory = '/data_drive/processed/freq_logging/'
    save_directory = f'{directory}top/'
    # Make sure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    # Load all files
    all_files_df = load_all_files(directory)
    # Calculate averages
    average_word_count, average_text_frequency = get_averages(all_files_df)
    categories = {
        'FnWCn': {'num_min_threshold': round(average_text_frequency), 'num_max_threshold': None, 'wc_min_threshold': 2, 'wc_max_threshold': None},
        'high_high': {'num_min_threshold': round(average_text_frequency), 'num_max_threshold': None, 'wc_min_threshold': round(average_word_count), 'wc_max_threshold': None},
        'low_low': {'num_min_threshold': 2, 'num_max_threshold': round(average_text_frequency), 'wc_min_threshold': 2, 'wc_max_threshold': round(average_word_count)},
        'low_high': {'num_min_threshold': 2, 'num_max_threshold': None, 'wc_min_threshold': round(average_word_count), 'wc_max_threshold': None},
    }
    # Step 2: Filtering using FnWCn and update ledger
    for category, params in categories.items():
        df_filtered = FnWCn(all_files_df, **params)
        # Save the DataFrame
        df_filtered.to_csv(os.path.join(save_directory, f'filtered_{category}.txt'), index=False)
        # Update ledger
        update_ledger(df_filtered, os.path.join(save_directory, f'top_{category}_ledger.txt'))
    # Combine all dataframes
    dataframes = [pd.read_csv(os.path.join(save_directory, f'filtered_{category}.txt')) for category in categories.keys()]
    final_df = pd.concat(dataframes, ignore_index=True)
    # Save 'text_clean' column to a text file
    final_df['text_clean'].drop_duplicates().to_csv(os.path.join(save_directory, 'text_clean.txt'), index=False, header=False)
    # Save the final combined dataframe
    final_df.to_csv(os.path.join(save_directory, 'final_top.txt'), index=False)

def dynamic_frequency_logging(df, column, directory = '/data_drive/processed/freq_logging/'):
    """
    This function takes a DataFrame and a column name, and identifies rows where the text 
    in the specified column appears more frequently than a dynamically determined freq_threshold and 
    the word count per row is more than a dynamically determined word_count_threshold. It logs these 
    high-frequency rows to separate text files, updating existing files or creating new ones.
    """
    # Ensure input is a DataFrame and column exists
    if not isinstance(df, pd.DataFrame) or column not in df.columns:
        print("Invalid input. Ensure you provide a DataFrame and an existing column name.")
        return None
    # Clean the DataFrame
    df = df[df[column].notna() & (df[column] != '')]
    # Compute word counts while ignoring whitespace and empty strings
    df = df.assign(word_count = df[column].str.split().apply(lambda x: len([word for word in x if word.strip() != ""])))
    # Count frequency of each text
    counter = Counter(df[column])
    # Make sure the directory exists, if not, create it
    directory = os.path.dirname(directory)
    os.makedirs(directory, exist_ok=True)
    # Load metadata if it exists, else create an empty dict
    try:
        with open(f'{directory}/metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
    # Get the unique frequency counts in descending order
    unique_freqs = sorted(set(counter.values()), reverse=True)
    # Iterate over the unique frequency counts
    for freq_threshold in unique_freqs:
        print(f"Processing frequency threshold: {freq_threshold}")
        # Identify high frequency texts
        high_freq_texts = [text for text, freq in counter.items() if freq == freq_threshold and text not in metadata]
        # Iterate over word counts from max to 1
        for word_count_threshold in range(df['word_count'].max(), 0, -1):
            # Filter DataFrame to only include high frequency rows with the specific word count
            df_high_freq = df[df[column].isin(high_freq_texts) & (df['word_count'] == word_count_threshold)]
            # If high frequency rows are found, save to file
            if not df_high_freq.empty:
                # Create an explicit copy to avoid SettingWithCopyWarning
                df_high_freq = df_high_freq.copy()
                # Remove duplicates
                df_high_freq.drop_duplicates(inplace=True)
                # Sort DataFrame by posted_date_time in descending order
                df_high_freq = df_high_freq.sort_values(by='posted_date_time', ascending=False)
                # Define file name
                file_name = os.path.join(directory, f"F{freq_threshold}WC{word_count_threshold}_log.txt")
                # Update metadata
                for text in df_high_freq[column].unique():
                    metadata[text] = file_name
                # Write the DataFrame to a new log file
                df_high_freq.to_csv(file_name, mode='w', header=True, index=False)
                # Define lower frequency file name
                lower_freq_file_name = os.path.join(directory, f"F{freq_threshold-1}WC{word_count_threshold}_log.txt")
                # Check if the lower frequency file exists and if it contains entries that have increased in frequency
                if os.path.exists(lower_freq_file_name):
                    # Load existing lower frequency file
                    df_lower_freq = pd.read_csv(lower_freq_file_name)
                    # Check if there are any entries that have increased in frequency
                    lower_freq_texts = df_lower_freq[df_lower_freq[column].isin(high_freq_texts)]
                    if not lower_freq_texts.empty:
                        print(f"Removing low frequency entries from {lower_freq_file_name}...")
                        # Remove entries that increased in frequency
                        df_lower_freq = df_lower_freq[~df_lower_freq[column].isin(high_freq_texts)]
                        # Write the updated DataFrame to the log file
                        df_lower_freq.to_csv(lower_freq_file_name, mode='w', header=True, index=False)
    # Save the metadata
    with open(f'{directory}/metadata.json', 'w') as f:
        json.dump(metadata, f)
    return None

def load_all_files(directory_path):
    """Load all txt files from the directory into a dataframe"""
    dataframes = []
    for file in os.listdir(directory_path):
        if file.endswith(".txt"):
            number, wc_part = file.split('F')[1].split('WC')
            wc = wc_part.split('_')[0]
            df = pd.read_table(os.path.join(directory_path, file), sep=',').sort_values(by='date')
            df['frequency'] = int(number)
            df['word_count'] = int(wc)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)

def FnWCn(df, num_min_threshold=None, num_max_threshold=None, wc_min_threshold=None, wc_max_threshold=None, top_percent=0.05):
    """Filter the dataframe based on frequency and word count thresholds and take the top percent of the remaining data"""
    df = df[(df['frequency'] >= num_min_threshold if num_min_threshold is not None else True) &
            (df['frequency'] <= num_max_threshold if num_max_threshold is not None else True) &
            (df['word_count'] >= wc_min_threshold if wc_min_threshold is not None else True) &
            (df['word_count'] <= wc_max_threshold if wc_max_threshold is not None else True)]
    total_entries = df['text_clean'].nunique()
    top_entries = int(total_entries * top_percent)
    top_frequent_df = df['text_clean'].value_counts().nlargest(top_entries).reset_index()
    top_frequent_df.columns = ['text_clean', 'count']
    return df[df['text_clean'].isin(top_frequent_df['text_clean'])]

def get_averages(df):
    """Calculate and return the average word count and frequency"""
    return df['word_count'].mean(), df['frequency'].mean()

def update_ledger(df, file_path):
    """Update the ledger file with the top 10 most recent entries"""
    grouped_df = df.groupby('text_clean')['posted_date_time'].agg(['min', 'max']).reset_index()
    grouped_df.columns = ['text_clean', 'first_date', 'last_date']
    top_10_recent = grouped_df.sort_values(by='last_date', ascending=False).head(10)
    if os.path.isfile(file_path):
        ledger_df = pd.read_csv(file_path)
        updated_df = pd.concat([ledger_df, top_10_recent]).drop_duplicates()
    else:
        updated_df = top_10_recent
    updated_df.to_csv(file_path, index=False)

# Utility function to load and combine top frequent text files
def load_and_combine_top(directory_path, num_min_threshold=None, num_max_threshold=None, wc_min_threshold=None, wc_max_threshold=None, top_percent=0.1):
    ''' 
    EDA Function - Examine text frequency and word count groups
    - High freq_threshold and low word_count_threshold signify low importance text.
    - High freq_threshold and high word_count_threshold indicate high importance text.
    - Low freq_threshold and high word_count_threshold imply high importance text.
    - Low freq_threshold and low word_count_threshold typically constitute common discussions, the importance of which may vary.
    '''
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
                df['frequency'] = int(number) 
                dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Getting the total number of unique 'text_clean' entries
    total_entries = combined_df['text_clean'].nunique()
    # Calculate how many entries we want to keep
    top_entries = int(total_entries * top_percent)
    # Get the top n% frequent text
    top_frequent_df = combined_df['text_clean'].value_counts().nlargest(top_entries).reset_index()
    top_frequent_df.columns = ['text_clean', 'count']
    # Return only the rows of top frequent text in the combined dataframe
    combined_df = combined_df[combined_df['text_clean'].isin(top_frequent_df['text_clean'])]
    return combined_df