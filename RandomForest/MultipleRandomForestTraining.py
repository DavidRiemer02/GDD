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
import time
from datetime import datetime
import csv

performance_dir = "performance"
os.makedirs(performance_dir, exist_ok=True)
train_log_path = os.path.join(performance_dir, "training_performance_log.csv")


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
            valid_cols = numeric_df.columns[numeric_df.notna().sum() >= 3]
            filtered_numeric_df = numeric_df[valid_cols]
            if not filtered_numeric_df.empty:
                skew_vals = skew(filtered_numeric_df, nan_policy='omit')
                mean_skew = np.nanmean(skew_vals)
                features['num_skew'] = np.clip(mean_skew, -10, 10)
            else:
                features['num_skew'] = 0
            filtered_numeric_df = numeric_df[valid_cols]
            if not filtered_numeric_df.empty:
                features['num_kurtosis'] = np.clip(np.nanmean(kurtosis(filtered_numeric_df, nan_policy='omit')), -10, 10)
            else:
                features['num_kurtosis'] = 0            
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
        
        # Ratio of numerical vs. categorical columns
        total_columns = df.shape[1]
        num_numerical = len(numeric_df.columns)
        num_categorical = len(categorical_df.columns)

        if total_columns > 0:
            features['num_vs_cat_ratio'] = num_numerical / (num_categorical + 1e-5)  # Avoid division by zero
        else:
            features['num_vs_cat_ratio'] = 0

        return pd.DataFrame([features])

    def extract_metanome_features(self, json_path):
        """Extract Metanome dependency features from JSON."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            num_columns = data.get("NumberOfColumns", 1)  # Avoid division by zero

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

    def train_multiple_models(self, real_data_folder, fake_data_folder, sample_size, n_estimators, max_depth, use_grid_search=False):
        """Train Random Forest models on real and fake datasets, iterating over subdirectories in metanomeResults."""
        feature_list = []

        for label, folder, meta_folder in [(1, real_data_folder, 'realData'), (0, fake_data_folder, 'fakeData')]:
            csv_files = glob(os.path.join(folder, "**", "*.csv"), recursive=True)

            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
                except Exception as e:
                    print(f"⚠️ Could not read {csv_file}: {e}")
                    continue

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

        start_time = time.time()

        if use_grid_search:
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [10, 25, 50],
                'min_samples_split': [2, 5, 10]
            }

            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_scaled, y)
            best_model = grid_search.best_estimator_

            model_name = "random_forest_grid_search.pkl"
            joblib.dump(best_model, os.path.join(self.model_dir, model_name))
            joblib.dump(scaler, os.path.join(self.model_dir, "scaler_grid_search.pkl"))
            print(f"GridSearchCV completed. Best model saved as '{model_name}'")
            print(f"🔎 Best Parameters: {grid_search.best_params_}")

            duration_ms = int((time.time() - start_time) * 1000)

            with open(train_log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow([
                        "Timestamp", "Use_GridSearch", "Sample_Size", "n_Estimators", "Max_Depth",
                        "Best_Params", "Train_Time_ms", "Model_Name"
                    ])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    use_grid_search,
                    sample_size,
                    grid_search.best_params_['n_estimators'],
                    grid_search.best_params_['max_depth'],
                    json.dumps(grid_search.best_params_),
                    duration_ms,
                    model_name
                ])

        else:
            model_name = f"random_forest_s{sample_size}_n{n_estimators}_d{max_depth}.pkl"
            rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf_classifier.fit(X_scaled, y)
            joblib.dump(rf_classifier, os.path.join(self.model_dir, model_name))
            joblib.dump(scaler, os.path.join(self.model_dir, f"scaler_s{sample_size}_n{n_estimators}_d{max_depth}.pkl"))
            print(f"Trained Random Forest (Samples={sample_size}, Trees={n_estimators}, Depth={max_depth})")

            duration_ms = int((time.time() - start_time) * 1000)

            with open(train_log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow([
                        "Timestamp", "Use_GridSearch", "Sample_Size", "n_Estimators", "Max_Depth",
                        "Best_Params", "Train_Time_ms", "Model_Name"
                    ])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    use_grid_search,
                    sample_size,
                    n_estimators,
                    max_depth,
                    "-",
                    duration_ms,
                    model_name
                ])


    import csv
    from datetime import datetime

    classification_log_path = os.path.join("performance", "classification_log.csv")
    os.makedirs("performance", exist_ok=True)

    def classify_new_datasets(self, base_folder, model_name="random_forest_grid_search.pkl"):
        """Classifies all datasets in a folder and logs classification metadata."""
        predictions = []

        try:
            if not os.path.exists(base_folder):
                print(f"Error: The folder '{base_folder}' does not exist.")
                return predictions

            # Identify data type
            if "realData" in base_folder:
                data_type = "realData"
            elif "fakeData" in base_folder:
                data_type = "fakeData"
            else:
                print(f"Could not determine dataset type for '{base_folder}'. Skipping...")
                return predictions

            csv_files = glob(os.path.join(base_folder, "**", "*.csv"), recursive=True)
            if not csv_files:
                print(f"No CSV datasets found in '{base_folder}'.")
                return predictions

            metanome_results_path = os.path.join(base_folder, "metanomeResults")
            print(f"Found {len(csv_files)} datasets in '{base_folder}'. Starting classification...")

            with open(self.classification_log_path, mode='a', newline='', encoding='utf-8') as log_file:
                writer = csv.writer(log_file)
                if log_file.tell() == 0:
                    writer.writerow([
                        "Timestamp", "Dataset", "Type", "Size_MB", "Rows", "Columns",
                        "Prediction", "Classification_Time_ms", "Model"
                    ])

                for csv_file in csv_files:
                    try:
                        start_time = time.time()

                        df = pd.read_csv(csv_file)
                        if df.empty:
                            print(f"Skipping empty dataset: {csv_file}")
                            continue

                        dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
                        json_path = self.find_metanome_json(dataset_name, metanome_results_path)

                        if not json_path:
                            print(f"No Metanome JSON for '{dataset_name}', skipping.")
                            continue

                        features = self.extract_combined_features(df, json_path)

                        model_path = os.path.join(self.model_dir, model_name)
                        scaler_path = os.path.join(self.model_dir, model_name.replace("random_forest", "scaler"))

                        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                            print(f"Model or scaler missing: {model_path} / {scaler_path}")
                            continue

                        model = joblib.load(model_path)
                        scaler = joblib.load(scaler_path)

                        X_scaled = scaler.transform(features)
                        prediction = model.predict(X_scaled)[0]
                        label = "real" if prediction == 1 else "fake"

                        predictions.append(label)

                        end_time = time.time()
                        duration_ms = int((end_time - start_time) * 1000)
                        file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
                        num_rows, num_cols = df.shape

                        # ✅ Write to CSV log
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            csv_file,
                            data_type,
                            f"{file_size_mb:.2f}",
                            num_rows,
                            num_cols,
                            label,
                            duration_ms,
                            model_name
                        ])

                        print(f"{csv_file} classified as {label} in {duration_ms} ms.")

                    except Exception as dataset_error:
                        print(f"⚠️ Error classifying {csv_file}: {dataset_error}")

        except Exception as e:
            print(f"❌ Error during classification: {e}")

        return predictions



if __name__ == "__main__":
    detector = GeneratedDatasetDetector()
    param_grid = [
        (500, 1, 20),
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

    print(f"Number of CSV datasets in realData: {real_csv_count}")
    print(f"Number of Metanome JSONs in realData: {real_json_count}")
    print(f"Total datasets in realData: {total_real_datasets}")

    print(f"Number of CSV datasets in fakeData: {fake_csv_count}")
    print(f"Number of Metanome JSONs in fakeData: {fake_json_count}")
    print(f"Total datasets in fakeData: {total_fake_datasets}")

    print(f"Total datasets across both: {total_real_datasets + total_fake_datasets}")

    detector.train_multiple_models("TrainingData/realData", "TrainingData/fakeData", 1000, 0, 0, use_grid_search=True)


    # Train models with different settings
    for sample_size, n_estimators, max_depth in param_grid:
        detector.train_multiple_models("TrainingData/realData", "TrainingData/fakeData", sample_size, n_estimators, max_depth)