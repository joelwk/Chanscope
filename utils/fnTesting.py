from sklearn.model_selection import ParameterGrid, train_test_split
from srcs.labeler import DialogDetector, apply_detector
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

def load_and_split_data(data_input, category_column, output_folder):
    X = data_input.drop(columns=[category_column])
    y = data_input[category_column]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25) # Same as above
    df_train = pd.DataFrame(X_train, columns=X.columns)
    df_train[category_column] = y_train
    df_val = pd.DataFrame(X_val, columns=X.columns)
    df_val[category_column] = y_val
    df_test = pd.DataFrame(X_test, columns=X.columns)
    df_test[category_column] = y_test
    # Save the training data
    save_data(df_train, output_folder, 'train.parquet')
    # Save the validation data
    save_data(df_val, output_folder, 'val.parquet')
    # Save the testing data
    save_data(df_test, output_folder, 'test.parquet')

def grid_search(data, param_grid):
    ''' Evluate the performance of the spam detector on a grid of parameters.'''
    results = []
    for params in ParameterGrid(param_grid):
        spam_ratio, labeled_data, similarity_training_data, spam_detector = apply_detector(data, **params)
        result = params.copy()
        result['spam_ratio'] = spam_ratio
        result['labeled_data'] = labeled_data
        results.append(result)
    return results