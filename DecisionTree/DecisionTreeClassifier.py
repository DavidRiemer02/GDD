import pandas as pd
import numpy as np
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis, entropy
from glob import glob

# ---- Feature Extraction Function ---- #
def extract_features(df):
    """Extracts statistical features from any dataset."""
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

    # Categorical Features
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        features['num_categorical'] = len(categorical_df.columns)
        features['cat_unique_ratio'] = categorical_df.nunique().mean()
        features['cat_mode_freq'] = categorical_df.mode().iloc[0].value_counts().mean()
        features['cat_entropy'] = entropy(categorical_df.apply(lambda x: x.value_counts(normalize=True), axis=0), nan_policy='omit').mean()

    return pd.DataFrame([features])

# ---- Train Decision Tree with Multiple Real Datasets ---- #
def train_decision_tree(real_data_folder, fake_data_file):
    """Trains a decision tree on multiple real datasets + fake dataset."""
    feature_list = []

    # Load & Process Multiple Real Datasets
    real_files = glob(os.path.join(real_data_folder, "*.csv"))  # All real datasets
    for file in real_files:
        df = pd.read_csv(file).head(1000)  # Use first 1000 rows
        real_features = extract_features(df)
        real_features["label"] = 1  # Real
        feature_list.append(real_features)

    # Load & Process Fake Data
    fake_df = pd.read_csv(fake_data_file).head(1000)
    fake_features = extract_features(fake_df)
    fake_features["label"] = 0  # Fake
    feature_list.append(fake_features)

    # Combine all extracted features
    feature_df = pd.concat(feature_list, ignore_index=True)

    # Prepare input features and labels
    X = feature_df.drop(columns=["label"])
    y = feature_df["label"]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Decision Tree
    dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=1337)
    dt_classifier.fit(X_scaled, y)

    # Save Model and Scaler
    joblib.dump(dt_classifier, "models/decisionTree/decision_tree_multi_real.pkl")
    joblib.dump(scaler, "models/decisionTree/scaler_multi_real.pkl")

    print(f"✅ Decision Tree trained on {len(real_files)} real datasets + Fake data!")

# ---- Classify Completely New Dataset ---- #
def classify_new_dataset(file_path):
    """Classifies a new dataset as Real or Fake based on extracted features."""
    new_df = pd.read_csv(file_path)

    # Extract features
    new_features = extract_features(new_df)

    # Load trained model and scaler
    dt_classifier = joblib.load("models/decisionTree/decision_tree_multi_real.pkl")
    scaler = joblib.load("models/decisionTree/scaler_multi_real.pkl")

    # Standardize features
    new_X_scaled = scaler.transform(new_features)

    # Predict
    prediction = dt_classifier.predict(new_X_scaled)
    label = "Real" if prediction[0] == 1 else "Fake"

    print(f"Classification Result: {label} for file {file_path}")

# ---- Example Usage ---- #
#Train with multiple real datasets (store them in "realData/" folder)
train_decision_tree("TrainingData/realData/", "TrainingData/fakeData/fake_dataset_1.csv")

#Classify a completely new dataset
classify_new_dataset("TestData/fakeData/Artificial_Data.csv")
print("Actual: Fake")
print("\n")

classify_new_dataset("TestData/fakeData/fake_dataset_3.csv")
print("Actual: Fake")
print("\n")

classify_new_dataset("TestData/fakeData/Generated_Dataset.csv")
print("Actual: Fake")
print("\n")


classify_new_dataset("TestData/realData/iris.csv")
print("Actual: Real")
print("\n")

classify_new_dataset("TestData/realData/uniprot_1001r_223c.csv")
print("Actual: Real")
print("\n")

classify_new_dataset("TestData/realData/CD.csv")
print("Actual: Real")
print("\n")

classify_new_dataset("TestData/realData/AnimalCrossingAccessories.csv")
print("Actual: Real")
print("\n")
