import pandas as pd
import json
from scipy.stats import skew, kurtosis, entropy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RandomForest.Utils.zipf_analysis import zipf_correlation as zc
from RandomForest.Utils.benford_analysis import benford_deviation as bd
from RandomForest.Utils.zipf_analysis import zipf_correlation as zc

def extract_features_from_csv(csv_file, json_file=None):
    """Extracts and prints statistical and dependency-based features from a dataset."""
    
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print("The dataset is empty.")
            return
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return
    
    features = {}

    # Numerical Features
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        features['num_mean'] = numeric_df.mean().mean()
        features['num_std'] = numeric_df.std().mean()
        features['num_min'] = numeric_df.min().mean()
        features['num_max'] = numeric_df.max().mean()
        features['num_skew'] = skew(numeric_df, nan_policy='omit').mean()
        features['num_kurtosis'] = kurtosis(numeric_df, nan_policy='omit').mean()
        features['benford_mae'] = bd(numeric_df.stack())
    else:
        features.update({key: 0 for key in ['num_mean', 'num_std', 'num_min', 'num_max', 
                                             'num_skew', 'num_kurtosis', 'benford_mae']})

    # Categorical Features
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        features['num_categorical'] = len(categorical_df.columns)
        features['cat_unique_ratio'] = categorical_df.nunique().mean()
        features['cat_mode_freq'] = categorical_df.mode().iloc[0].value_counts().mean()
        features['cat_entropy'] = entropy(categorical_df.apply(lambda x: x.value_counts(normalize=True), axis=0), nan_policy='omit').mean()
        features['zipf_corr'] = zc(categorical_df.stack())
    else:
        features.update({key: 0 for key in ['num_categorical', 'cat_unique_ratio', 'cat_mode_freq', 
                                             'cat_entropy', 'zipf_corr']})
    total_columns = df.shape[1]
    num_numerical = len(numeric_df.columns)
    num_categorical = len(categorical_df.columns)

    if total_columns > 0:
        features['num_vs_cat_ratio'] = num_numerical / (num_categorical + 1e-5)  # Avoid division by zero
    else:
        features['num_vs_cat_ratio'] = 0
    

    # Metanome Dependency-Based Features (if JSON file is provided)
    if json_file:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            num_columns = data.get("NumberOfColumns", 1)  # Avoid division by zero

            features.update({
                "fds_ratio": data.get("FDs_count", 0) / num_columns,
                "uccs_ratio": data.get("UCCs_count", 0) / num_columns,
                "inds_ratio": data.get("INDs_count", 0) / num_columns,
                "max_fd_length_norm": data.get("Max_FD_Length", 0) / num_columns,
                "fds_count": data.get("FDs_count", 0),
                "uccs_count": data.get("UCCs_count", 0),
                "inds_count": data.get("INDs_count", 0),
                "max_fd_length": data.get("Max_FD_Length", 0)
            })
        except Exception as e:
            print(f"Error reading Metanome JSON file: {e}")

    # Print extracted features
    print("\n🔍 Extracted Features:")
    for key, value in features.items():
        print(f"{key}: {value:.6f}")

csv_path = "TestData/realData/titanic.csv"
json_path = "TestData/realData/MetanomeResults/titanic_Results.json"  # Optional
extract_features_from_csv(csv_path, json_path)