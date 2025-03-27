import os
import sys
import subprocess
import glob
from RandomForest.Utils.cleanCSV import clean_csv_quotes
from RandomForest.MultipleRandomForestTraining import GeneratedDatasetDetector
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "RandomForest")))

# ---- Configuration ---- #
java_exe = "C:\\Users\\David\\.jdks\\openjdk-18.0.2.1\\bin\\java"  # Full path to Java
test_base_dir_real = "TestData/realData"  
test_base_dir_fake = "TestData/fakeData"
metanome_jar = "generatedDatasetDetector.jar"  # JAR file path
java_memory = "-Xmx32G"  # Adjust memory as needed

# Results directory
test_result_dir_real = os.path.join(test_base_dir_real, "metanomeResults")
test_result_dir_fake = os.path.join(test_base_dir_fake, "metanomeResults")


# ---- Helper Functions ---- #
def clean_csv_in_place(file_path):
    """Cleans a CSV file in place by removing problematic quotes."""
    temp_output = file_path + ".tmp"
    clean_csv_quotes(file_path, temp_output)  # Call with both input and output paths
    os.replace(temp_output, file_path)  # Overwrite original file
    print(f"✅ Cleaned in-place: {file_path}")


def clean_all_csv_files(directory):
    """Recursively cleans all CSV files in a directory and its subdirectories."""
    print(f"🔍 Cleaning CSV files in {directory} ...")
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    
    for csv_file in csv_files:
        print(f"📂 Cleaning {csv_file} ...")
        clean_csv_in_place(csv_file)


import os
import time
import csv
import subprocess
import pandas as pd
from datetime import datetime

# Define performance logging directory and file
performance_dir = "performance"
os.makedirs(performance_dir, exist_ok=True)
performance_log_path = os.path.join(performance_dir, "metanome_performance_log.csv")

# Define timestamp formatter
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_metanome_if_needed(dataset_path, result_dir):
    """Run Metanome only if output does not already exist. Also logs performance info."""
    base_name = os.path.basename(dataset_path).replace(".csv", "")
    relative_path = os.path.relpath(os.path.dirname(dataset_path), start=os.path.dirname(result_dir))
    metanome_result_folder = os.path.join(result_dir, relative_path)
    os.makedirs(metanome_result_folder, exist_ok=True)

    result_file = os.path.join(metanome_result_folder, f"{base_name}_Results.json")

    if os.path.exists(result_file):
        print(f"[{timestamp()}] ✅ Metanome result already exists: {result_file}, skipping.")
        return

    try:
        file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        df = pd.read_csv(dataset_path, nrows=5)
        num_columns = df.shape[1]
    except Exception as e:
        print(f"[{timestamp()}] ⚠️ Failed to read {dataset_path} for metadata: {e}")
        return

    # Run Metanome and log time
    print(f"[{timestamp()}] 🚀 Running Metanome on: {dataset_path}")
    start_time = time.time()
    command = [
        java_exe, java_memory, "-jar", metanome_jar,
        "--input-file", dataset_path,
        "--output-file", result_file
    ]

    try:
        subprocess.run(command, check=True)
        duration_ms = int((time.time() - start_time) * 1000)

        # Write performance log
        with open(performance_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Timestamp", "Dataset", "Size_MB", "Columns", "Metanome_Time_ms", "Output"])
            writer.writerow([
                timestamp(), dataset_path, f"{file_size_mb:.2f}", num_columns, duration_ms, result_file
            ])

        print(f"[{timestamp()}] ✅ Metanome completed in {duration_ms} ms")

    except subprocess.CalledProcessError as e:
        print(f"[{timestamp()}] ❌ Error running Metanome on {dataset_path}: {e}")


def get_all_csv_files(base_dir):
    return glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)


def test_pipeline():
    print("Starting full test pipeline with metrics...")

    for base_dir, result_dir in [(test_base_dir_real, test_result_dir_real),
                                 (test_base_dir_fake, test_result_dir_fake)]:
        os.makedirs(result_dir, exist_ok=True)
        clean_all_csv_files(base_dir)
        for csv_file in get_all_csv_files(base_dir):
            run_metanome_if_needed(csv_file, result_dir)

    detector = GeneratedDatasetDetector()

    # Get predictions only (no probabilities)
    real_preds = detector.classify_new_datasets(test_base_dir_real)
    fake_preds = detector.classify_new_datasets(test_base_dir_fake)

    # Ground truth and predictions
    y_true = [1] * len(real_preds) + [0] * len(fake_preds)  # 1 = real, 0 = fake
    y_pred = [1 if p == "real" else 0 for p in real_preds + fake_preds]

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n Classification Report:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"\n Confusion Matrix:\n{cm}")

    #Visualize confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_pipeline()