from sklearn.model_selection import ParameterGrid, train_test_split
from srcs.labeler import DialogDetector, apply_detector
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from utils import fnTesting, fnPlots
from kneed import KneeLocator
import matplotlib.pyplot as plt

import os
import pandas as pd
def test_roc(data):
    # Assuming spam_label_source is numerical
    X = data[['spam_label_source']].values
    # If spam_label is categorical, you may need to encode it
    le = LabelEncoder()
    y = le.fit_transform(data['spam_label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score for Classifier: {roc_auc}")
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Classifier (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Baseline ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
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

def evaluate_spam_threshold(data, param_grid):
    # Perform grid search on the data using the specified parameters
    results = fnTesting.grid_search(data, param_grid)
    summary_table = fnPlots.create_summary_table(results)

    # Plot the proportion of spam against similarity threshold using matplotlib
    plt.plot(summary_table['similarity_threshold'], summary_table['spam_ratio'], marker='o')
    plt.xlabel('Threshold')
    plt.ylabel('Proportion of Spam')
    plt.title('Spam Proportion vs Threshold')

    # Find the knee/elbow in the curve using the KneeLocator library
    kneedle = KneeLocator(summary_table['similarity_threshold'], summary_table['spam_ratio'], curve='convex', direction='decreasing')
    plt.vlines(kneedle.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.show()

    # Print the optimal threshold for a significant drop-off, based on the analysis
    optimal_threshold = kneedle.knee
    print(f"Optimal threshold for significant drop-off: {optimal_threshold}")
    
    return optimal_threshold, results ,summary_table

def test_roc(data):
    # Assuming spam_label_source is numerical
    X = data[['spam_label_source']].values
    # If spam_label is categorical, you may need to encode it
    le = LabelEncoder()
    y = le.fit_transform(data['spam_label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC Score for Classifier: {roc_auc}")
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'Classifier (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Baseline ROC Curve')
    plt.legend(loc="lower right")
    plt.show()