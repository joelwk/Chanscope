import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

def plot_kde(data):
    data = data.copy()
    # Extract hour and day from 'posted_date_time'
    data.loc[:, 'hour'] = data['posted_date_time'].dt.hour
    data.loc[:, 'day'] = data['posted_date_time'].dt.day
    data.loc[:, 'month'] = data['posted_date_time'].dt.month
    # Plot distribution of rows by hour

    # Create a figure and a grid of subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))

    # Plot distribution of rows by hour
    sns.kdeplot(data['hour'], fill=True, ax=ax[0])
    ax[0].set_title('Distribution of Rows by Hour')
    ax[0].set_xlabel('Hour')
    ax[0].set_ylabel('Density')

    # Plot distribution of rows by day
    sns.kdeplot(data['day'], fill=True, ax=ax[1])
    ax[1].set_title('Distribution of Rows by Day')
    ax[1].set_xlabel('Day')
    ax[1].set_ylabel('Density')

    # Plot distribution of rows by month
    sns.kdeplot(data['month'], fill=True, ax=ax[2])
    ax[2].set_title('Distribution of Rows by Month')
    ax[2].set_xlabel('Month')
    ax[2].set_ylabel('Density')

    # Display the figure with subplots
    plt.tight_layout()
    plt.show()

def plot_hist(data):
    data = data.copy()
    # Check if 'posted_date_time' column exists in data
    if 'posted_date_time' not in data.columns:
        print("Column 'posted_date_time' not found in data.")
    else:
        # Extract hour, day, and month from 'posted_date_time'
        data.loc[:, 'hour'] = data['posted_date_time'].dt.hour
        data.loc[:, 'day'] = data['posted_date_time'].dt.day
        data.loc[:, 'month'] = data['posted_date_time'].dt.month

        # Create a figure and a grid of subplots
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6))

        # Plot distribution of rows by hour
        sns.histplot(data['hour'], kde=True, bins=24, ax=ax[0, 0])
        ax[0, 0].set_title('Density Plot for Hour')
        ax[0, 0].set_xlabel('Hour')
        ax[0, 0].set_ylabel('Density')

        # Plot distribution of rows by day
        sns.histplot(data['day'], kde=True, bins=31, ax=ax[0, 1])
        ax[0, 1].set_title('Density Plot for Day')
        ax[0, 1].set_xlabel('Day')
        ax[0, 1].set_ylabel('Density')

        # Plot distribution of rows by month
        sns.histplot(data['month'], kde=True, bins=12, ax=ax[1, 0])
        ax[1, 0].set_title('Density Plot for Month')
        ax[1, 0].set_xlabel('Month')
        ax[1, 0].set_ylabel('Density')

        # Check the number of unique threads in the sample
        unique_threads = data['thread_id'].nunique()
        print(f"Number of unique threads in the sample: {unique_threads}")

        # Check the distribution of posts per thread
        posts_per_thread = data['thread_id'].value_counts()
        sns.histplot(posts_per_thread, kde=False, bins=30, ax=ax[1, 1])
        ax[1, 1].set_title('Distribution of Posts per Thread')
        ax[1, 1].set_xlabel('Number of Posts')
        ax[1, 1].set_ylabel('Number of Threads')

        # Display the figure with subplots
        plt.tight_layout()
        plt.show()

def profile_date(df, date_col):
    df = df.copy()
    # Convert date_col to datetime if it's not already
    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d %H:%M:%S')
    # Print the minimum and maximum dates
    print(f"Minimum date: {df[date_col].min()}")
    print(f"Maximum date: {df[date_col].max()}")
    # Create a histogram of the dates
    df[date_col].hist(bins=50, figsize=(10, 5))
    plt.xlabel('Date')
    plt.ylabel('Frequency')
    plt.title('Distribution of Dates')
    plt.show()
    
class FrequencyLoggerPlotter:
    def __init__(self, df, column):
        self.df = df
        self.column = column
        self.counter = Counter(self.df[self.column])

    def frequency_logging(self, freq_threshold, word_count_threshold):
        high_freq_texts = [text for text, freq in self.counter.items() if freq > freq_threshold]
        df_high_freq = self.df[self.df[self.column].isin(high_freq_texts)]
        df_high_freq = df_high_freq[df_high_freq[self.column].str.split().str.len() > word_count_threshold]
        return len(df_high_freq)

    def plot_threshold_distribution(self, max_freq, max_step, step_size):
        counts = []
        labels = []
    
        for freq_threshold in range(max_freq, max_step, -step_size):
            for word_count_threshold in range(max_freq, max_step, -step_size):
                count = self.frequency_logging(freq_threshold, word_count_threshold)
                if count > 0:
                    counts.append(count)
                    labels.append(f'F={freq_threshold}, WC={word_count_threshold}')

        plt.figure(figsize=(20, 10))
        plt.plot(labels, counts, marker='o')
        plt.xlabel('Thresholds')
        plt.ylabel('Number of High-Frequency Rows')
        plt.title('Distribution of Frequency and Word Count Thresholds')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_threshold_3Dscatter_distribution(self, max_freq, max_step, step_size):
        X, Y, Z = [], [], []

        for freq_threshold in range(max_freq, max_step, -step_size):
            for word_count_threshold in range(max_freq, max_step, -step_size):
                count = self.frequency_logging(freq_threshold, word_count_threshold)
                if count > 0:
                    X.append(freq_threshold)
                    Y.append(word_count_threshold)
                    Z.append(count)

        fig = plt.figure(figsize=(12, 15))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X, Y, Z, c=Z, cmap='viridis')
        ax.set_xlabel('Frequency Threshold')
        ax.set_ylabel('Word Count Threshold')
        ax.set_zlabel('Number of High-Frequency Rows')
        plt.title('3D Scatter Plot of Frequency and Word Count Thresholds')
        plt.show()


