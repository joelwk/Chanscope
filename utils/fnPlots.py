import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D

def extract_date_parts(data, column_name):
    data['hour'] = pd.to_datetime(data[column_name]).dt.hour
    data['day'] = pd.to_datetime(data[column_name]).dt.day
    data['month'] = pd.to_datetime(data[column_name]).dt.month

def plot_kde(data, w, h):
    if 'posted_date_time' not in data.columns:
        print("Column 'posted_date_time' not found in data.")
    else:
        extract_date_parts(data, 'posted_date_time')
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(w, h))
        sns.kdeplot(data['hour'], fill=True, ax=ax[0])
        sns.kdeplot(data['day'], fill=True, ax=ax[1])
        sns.kdeplot(data['month'], fill=True, ax=ax[2])
        for i, label in enumerate(['Hour', 'Day', 'Month']):
            ax[i].set_title(f'Distribution of Rows by {label}')
            ax[i].set_xlabel(label)
            ax[i].set_ylabel('Density')
        plt.tight_layout()
        plt.show()

def plot_hist(data, w, h):
    if 'posted_date_time' not in data.columns:
        print("Column 'posted_date_time' not found in data.")
        return

    extract_date_parts(data, 'posted_date_time')
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(w, h))

    # Plot distribution of rows by hour
    sns.histplot(data['hour'], kde=True, bins=24, ax=ax[0])
    ax[0].set_title('Density Plot for Hour')
    ax[0].set_xlabel('Hour')
    ax[0].set_ylabel('Density')

    # Plot distribution of rows by day
    sns.histplot(data['day'], kde=True, bins=31, ax=ax[1])
    ax[1].set_title('Density Plot for Day')
    ax[1].set_xlabel('Day')
    ax[1].set_ylabel('Density')

    # Plot distribution of rows by month
    sns.histplot(data['month'], kde=True, bins=12, ax=ax[2])
    ax[2].set_title('Density Plot for Month')
    ax[2].set_xlabel('Month')
    ax[2].set_ylabel('Density')
    # Display the figure with subplots
    plt.tight_layout()
    plt.show()

def profile_date_plot(df, date_col):
    df = df.copy()
    # Convert date_col to datetime
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

def profile_date(data, date_column):
    # Define your profiling function here
    # This is just a placeholder for demonstration purposes
    profile_report = data[date_column].describe()
    return profile_report

# Class for logging and plotting frequency distributions
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
        
def create_summary_table(results):
    summary_data = []
    for res in results:
        summary_row = res.copy()
        summary_row['spam_ratio'] = summary_row['spam_ratio']['SPAM'] # Assuming spam_ratio is a dictionary
        summary_data.append(summary_row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.drop(columns='labeled_data', inplace=True)
    return summary_df

custom_palette = {'SPAM': 'green', 'NOT_SPAM': 'blue','UNASSIGNED':'yellow','ERROR':'red'}
def plot_histograms(summary_table):
    num_results = len(summary_table)
    fig, axes = plt.subplots(3, num_results, figsize=(16, 8))

    for idx, result in enumerate(summary_table):
        label_data = result['labeled_data']

        # Plot SPAM/NOT_SPAM distribution with hue
        sns.histplot(label_data, x='spam_label', hue='spam_label', palette=custom_palette, kde=False, discrete=True, ax=axes[0, idx], bins=2)
        axes[0, idx].set_title(f"Classification: {result['similarity_threshold']} S/NS", fontsize=10)
        axes[0, idx].set_xticks([0, 1])
        axes[0, idx].set_xticklabels(['NOT_SPAM', 'SPAM'], fontsize=8)

        # Plot word lengths horizontally with hue
        word_lengths = label_data['text_clean'].apply(lambda x: len(str(x).split()))
        sns.histplot(label_data, y=word_lengths, hue='spam_label', palette=custom_palette, kde=False, ax=axes[1, idx])
        axes[1, idx].set_title(f"Word lengths: {result['similarity_threshold']} - WL", fontsize=10)

        # Plot sentence lengths horizontally with hue
        sentence_lengths = label_data['text_clean'].apply(lambda x: len(str(x)))
        sns.histplot(label_data, y=sentence_lengths, hue='spam_label', palette=custom_palette, kde=False, ax=axes[2, idx])
        axes[2, idx].set_title(f"Sentence lengths: {result['similarity_threshold']} - S/L", fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_scatter_charts(summary_table):
    fig, axes = plt.subplots(1, len(summary_table), figsize=(12, 5))
    
    for idx, result in enumerate(summary_table):
        label_data = result['labeled_data']
        # Calculate word counts
        word_counts = label_data['text_clean'].apply(lambda x: len(str(x).split()))
        # Calculate sequence lengths
        seq_lengths = label_data['text_clean'].apply(lambda x: len(str(x)))
        # Plot scatter chart with spam ratio as hue
        sns.scatterplot(x=word_counts, y=seq_lengths, hue=label_data['spam_label'], palette=custom_palette, ax=axes[idx])
        axes[idx].set_title(f"Scatter: {result['similarity_threshold']} WC/SL", fontsize=8)
        axes[idx].set_xlabel('Word Count')
        axes[idx].set_ylabel('Sequence Length')
    
    plt.tight_layout()
    plt.show()

# Plot the accuracy and loss
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Plot training and validation accuracy
    ax1 = axes[0]
    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_title('Training and Validation Accuracy')
    ax1.grid(True)

    ax1_val = ax1.twinx() # Create a secondary y-axis
    ax1_val.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax1_val.set_ylabel('Validation Accuracy', color='red')
    ax1_val.tick_params(axis='y', labelcolor='red')

    # Plot training and validation loss
    ax2 = axes[1]
    ax2.plot(history.history['loss'], label='Training Loss', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Loss', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_title('Training and Validation Loss')
    ax2.grid(True)

    ax2_val = ax2.twinx() # Create a secondary y-axis
    ax2_val.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax2_val.set_ylabel('Validation Loss', color='red')
    ax2_val.tick_params(axis='y', labelcolor='red')

    plt.tight_layout()
    plt.show()
