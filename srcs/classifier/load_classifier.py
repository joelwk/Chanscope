from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import os
import pandas as pd

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(df, output_folder, filename):
    ensure_directory_exists(output_folder)
    file_path = os.path.join(output_folder, filename)
    df.to_parquet(file_path)

def load_and_split_data_smote(data_input, category_column, output_folder):
    # Assuming data_input is a DataFrame
    X = data_input.drop(columns=[category_column])
    y = data_input[category_column]
    # Split the data into training, validation, and testing sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25) # Splitting remaining 80% into 60% train and 20% validation
    # Resample the training data
    ros = RandomOverSampler(sampling_strategy='auto') # Adjust as needed
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    # Combine resampled X and y back into training DataFrame
    df_resampled_train = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled_train[category_column] = y_resampled
    # Combine X_val and y_val back into validation DataFrame
    df_val = pd.DataFrame(X_val, columns=X.columns)
    df_val[category_column] = y_val
    # Combine X_test and y_test back into testing DataFrame
    df_test = pd.DataFrame(X_test, columns=X.columns)
    df_test[category_column] = y_test
    # Save the resampled training data
    save_data(df_resampled_train, output_folder, 'train.parquet')
    # Save the validation data
    save_data(df_val, output_folder, 'val.parquet')
    # Save the testing data
    save_data(df_test, output_folder, 'test.parquet')