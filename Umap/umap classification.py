from UMAPColumnEmbedder_Advanced import UMAPColumnEmbedder
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# === Load model and embedder ===
embedder = UMAPColumnEmbedder()
embedder.load_model()
clf = joblib.load("models/umapRandomForest/random_forest_umap.pkl")

# === Classify all CSV files and compute accuracy ===
true_labels = []
predicted_labels = []

test_data_root = "TestData"

for dirpath, _, filenames in os.walk(test_data_root):
    for file_name in filenames:
        if not file_name.endswith(".csv"):
            continue
        file_path = os.path.join(dirpath, file_name)
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
            numeric_df = df.select_dtypes(include=["int64", "float64"])
            emb = embedder.transform_and_aggregate(numeric_df).reshape(1, -1)
            prediction = clf.predict(emb)[0]
            predicted_labels.append(prediction)

            # Infer ground truth from folder name
            label = 1 if "realData" in dirpath else 0
            true_labels.append(label)

            label_str = "Real" if prediction == 1 else "Fake"
            print(f"{file_path}: {label_str}")
        except Exception as e:
            print(f"Failed to classify {file_path}: {e}")

# === Compute and print accuracy ===
if true_labels and predicted_labels:
    acc = accuracy_score(true_labels, predicted_labels)
    print(f"\nClassification Accuracy on TestData: {acc:.4f}")
else:
    print("No valid classifications performed.")
