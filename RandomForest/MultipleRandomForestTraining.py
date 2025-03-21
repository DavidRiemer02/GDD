import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy
from glob import glob
from RandomForest.Utils.benford_analysis import benford_deviation as bd
from RandomForest.Utils.zipf_analysis import zipf_correlation as zc
from sklearn.model_selection import GridSearchCV



class GeneratedDatasetDetector:
    def __init__(self):
        """Initialize directories and model storage."""
        self.model_dir = "models/randomForest"
        os.makedirs(self.model_dir, exist_ok=True)

    def extract_features(self, df):
        """Extracts statistical features from a dataset."""
        features = {}
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        if not numeric_df.empty:
            features['num_mean'] = numeric_df.mean().mean()
            features['num_std'] = numeric_df.std().mean()
            features['num_min'] = numeric_df.min().mean()
            features['num_max'] = numeric_df.max().mean()
            features['num_skew'] = np.clip(skew(numeric_df, nan_policy='omit').mean(), -10, 10)
            features['num_kurtosis'] = np.clip(kurtosis(numeric_df, nan_policy='omit').mean(), -10, 10)
            features['benford_mae'] = bd(numeric_df.stack())
        else:
            features.update(dict.fromkeys(['num_mean', 'num_std', 'num_min', 'num_max',
                                           'num_skew', 'num_kurtosis', 'benford_mae'], 0))

        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            features['num_categorical'] = len(categorical_df.columns)
            features['cat_unique_ratio'] = categorical_df.nunique().mean()
            features['cat_mode_freq'] = categorical_df.mode().iloc[0].value_counts().mean()
            features['cat_entropy'] = entropy(categorical_df.apply(lambda x: x.value_counts(normalize=True), axis=0), nan_policy='omit').mean()
            features['zipf_corr'] = zc(categorical_df.stack())
        else:
            features.update(dict.fromkeys(['num_categorical', 'cat_unique_ratio', 'cat_mode_freq', 'cat_entropy', 'zipf_corr'], 0))

        return pd.DataFrame([features])

    def extract_metanome_features(self, json_path):
        """Extract Metanome dependency features from JSON."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            num_columns = data.get("NumberOfColumns", 1)  # Avoid division by zero

            scaling_factor = 5  # Adjust weight for importance


            return {
                "fds_ratio": data.get("FDs_count", 0) / num_columns,
                "uccs_ratio": data.get("UCCs_count", 0) / num_columns,
                "inds_ratio": data.get("INDs_count", 0) / num_columns,
                "max_fd_length_norm": data.get("Max_FD_Length", 0) / num_columns,
                "fds_count": data.get("FDs_count", 0),
                "uccs_count": data.get("UCCs_count", 0),
                "inds_count": data.get("INDs_count", 0),
                "max_fd_length": data.get("Max_FD_Length", 0)
            }
        except Exception as e:
            print(f"⚠️ Error reading Metanome JSON {json_path}: {e}")
            return dict.fromkeys(['fds_ratio', 'uccs_ratio', 'inds_ratio', 'max_fd_length_norm',
                                  'fds_count', 'uccs_count', 'inds_count', 'max_fd_length'], 0)

    def extract_combined_features(self, df, json_path):
        """Combine statistical and Metanome dependency features."""
        stat_features = self.extract_features(df)
        meta_features = self.extract_metanome_features(json_path)
        return stat_features.assign(**meta_features)

    def find_metanome_json(self, dataset_name, meta_folder):
        """Find the correct Metanome JSON for a given dataset."""
        direct_path = os.path.join(meta_folder, f"{dataset_name}_Results.json")
        subfolder_paths = glob(os.path.join(meta_folder, "*", f"{dataset_name}_Results.json"))

        if os.path.exists(direct_path):
            return direct_path
        elif subfolder_paths:
            return subfolder_paths[0]
        return None

    def train_multiple_models(self, real_data_folder, fake_data_folder, sample_size, n_estimators, max_depth):
        """Train Random Forest models on real and fake datasets, iterating over subdirectories in metanomeResults."""
        feature_list = []

        for label, folder, meta_folder in [(1, real_data_folder, 'realData'), (0, fake_data_folder, 'fakeData')]:
            csv_files = glob(os.path.join(folder, "**", "*.csv"), recursive=True)

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)

                if len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)

                dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
                metanome_results_path = os.path.join(folder, "metanomeResults")

                json_path = self.find_metanome_json(dataset_name, metanome_results_path)

                if json_path:
                    combined_features = self.extract_combined_features(df, json_path)
                    combined_features["label"] = label
                    feature_list.append(combined_features)
                else:
                    print(f"⚠️ No Metanome JSON found for {dataset_name}. Skipping...")

        if not feature_list:
            print("⚠️ No valid datasets found! Training aborted.")
            return

        feature_df = pd.concat(feature_list, ignore_index=True)
        X = feature_df.drop(columns=["label"])
        y = feature_df["label"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_classifier.fit(X_scaled, y)

        joblib.dump(rf_classifier, os.path.join(self.model_dir, f"random_forest_s{sample_size}_n{n_estimators}_d{max_depth}.pkl"))
        joblib.dump(scaler, os.path.join(self.model_dir, f"scaler_s{sample_size}_n{n_estimators}_d{max_depth}.pkl"))
        print(f"✅ Trained Random Forest (Samples={sample_size}, Trees={n_estimators}, Depth={max_depth})")

    def classify_new_datasets(self, base_folder):
        """Classifies all datasets in a given folder (TestData/realData or TestData/fakeData)."""
        try:
            if not os.path.exists(base_folder):
                print(f"❌ Error: The folder '{base_folder}' does not exist.")
                return

            # Identify data type (realData or fakeData)
            if "realData" in base_folder:
                data_type = "realData"
            elif "fakeData" in base_folder:
                data_type = "fakeData"
            else:
                print(f"⚠️ Could not determine dataset type (realData/fakeData) for '{base_folder}'. Skipping...")
                return

            # Find all CSV datasets in the provided folder (including subfolders)
            csv_files = glob(os.path.join(base_folder, "**", "*.csv"), recursive=True)
            if not csv_files:
                print(f"⚠️ No CSV datasets found in '{base_folder}'.")
                return

            # Find Metanome results folder
            metanome_results_path = os.path.join(base_folder, "metanomeResults")

            print(f"🔍 Found {len(csv_files)} datasets in '{base_folder}'. Starting classification...")

            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                if df.empty:
                    print(f"⚠️ The dataset '{csv_file}' is empty and cannot be classified.")
                    continue

                dataset_name = os.path.splitext(os.path.basename(csv_file))[0]

                # Find corresponding Metanome JSON
                json_path = self.find_metanome_json(dataset_name, metanome_results_path)

                if not json_path:
                    print(f"⚠️ No Metanome JSON found for '{dataset_name}'. Skipping...")
                    continue

                # Extract features and classify
                features = self.extract_combined_features(df, json_path)
                model_files = glob(os.path.join(self.model_dir, "random_forest_*.pkl"))
                results = {}

                for model_file in model_files:
                    model_name = os.path.basename(model_file).replace(".pkl", "")
                    scaler_file = model_file.replace("random_forest", "scaler")
                    model = joblib.load(model_file)
                    scaler = joblib.load(scaler_file)
                    X_scaled = scaler.transform(features)
                    prediction = model.predict(X_scaled)[0]
                    label = "Real" if prediction == 1 else "Fake"
                    results[model_name] = label

                # Print classification results
                print(f"✅ Classification Results for {csv_file}:")
                for model, label in results.items():
                    print(f"  {model}: {label}")

        except Exception as e:
            print(f"⚠️ Error during classification: {e}")



if __name__ == "__main__":
    detector = GeneratedDatasetDetector()
    param_grid = [
        (500, 25, 5),
        (500, 100, 10),    # Small dataset setup
        (2000, 500, 20),   # Medium dataset setup
        (5000, 1000, 50),  # Large dataset setup
        (10000, 2000, 100) # Very large dataset setup
    ]
    def count_datasets(base_folder):
        """Counts all datasets in a folder, including CSVs and JSONs inside metanomeResults."""
    
        # Count CSV files (actual datasets)
        csv_files = glob(os.path.join(base_folder, "**", "*.csv"), recursive=True)

        # Count JSON files (Metanome results)
        json_files = glob(os.path.join(base_folder, "metanomeResults", "**", "*.json"), recursive=True)

        return len(csv_files), len(json_files)

    # Count datasets in realData and fakeData
    real_csv_count, real_json_count = count_datasets("TrainingData/realData")
    fake_csv_count, fake_json_count = count_datasets("TrainingData/fakeData")

    # Total datasets (CSV + JSON)
    total_real_datasets = real_csv_count + real_json_count
    total_fake_datasets = fake_csv_count + fake_json_count

    print(f"📂 Number of CSV datasets in realData: {real_csv_count}")
    print(f"📂 Number of Metanome JSONs in realData: {real_json_count}")
    print(f"📊 Total datasets in realData: {total_real_datasets}")

    print(f"📂 Number of CSV datasets in fakeData: {fake_csv_count}")
    print(f"📂 Number of Metanome JSONs in fakeData: {fake_json_count}")
    print(f"📊 Total datasets in fakeData: {total_fake_datasets}")

    print(f"Total datasets across both: {total_real_datasets + total_fake_datasets}")

    # Train models with different settings
    for sample_size, n_estimators, max_depth in param_grid:
        detector.train_multiple_models("TrainingData/realData", "TrainingData/fakeData", sample_size, n_estimators, max_depth)