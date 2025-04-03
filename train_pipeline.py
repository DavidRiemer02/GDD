import os
import subprocess
import glob
import sys
import pandas as pd
from datetime import datetime
from RandomForest.Utils.readlargeData import read_large_data

# ---- Configuration ---- #
java_exe = "C:\\Users\\David\\.jdks\\openjdk-18.0.2.1\\bin\\java"
base_dir = "TrainingData"
data_types = ["fakeData", "realData"]
metanome_jar = "NoteBook_UI/generatedDatasetDetector.jar"
java_memory = "-Xmx32G"

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "RandomForest")))

# ---- Helper Functions ---- #
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_csv_in_place(file_path):
    temp_output = file_path + ".tmp"
    read_large_data(file_path, temp_output)

    if os.path.exists(temp_output):
        os.replace(temp_output, file_path)
    else:
        print(f"Skipped replacing {file_path}: cleaning failed or no output produced.")


def clean_all_csv_files(directory):
    print(f"[{timestamp()}] Cleaning CSV files in {directory} ...")
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)

    for csv_file in csv_files:
        try:
            print(f"[{timestamp()}] Cleaning {csv_file} ...")
            clean_csv_in_place(csv_file)
        except Exception as e:
            print(f"Error cleaning {csv_file}: {e}")


import time
import csv

performance_dir = "performance"
os.makedirs(performance_dir, exist_ok=True)
performance_log_path = os.path.join(performance_dir, "metanome_performance_log.csv")

def run_metanome_if_needed(dataset_path, result_dir):
    base_name = os.path.basename(dataset_path).replace(".csv", "")
    relative_path = os.path.relpath(os.path.dirname(dataset_path), start=os.path.dirname(result_dir))
    metanome_result_folder = os.path.join(result_dir, relative_path)
    os.makedirs(metanome_result_folder, exist_ok=True)

    result_file = os.path.join(metanome_result_folder, f"{base_name}_Results.json")

    if os.path.exists(result_file):
        return  # Skip if already exists

    try:
        file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        df = pd.read_csv(dataset_path, nrows=5)
        num_columns = df.shape[1]
    except Exception as e:
        print(f"[{timestamp()}] Failed to read {dataset_path} for metadata: {e}")
        return

    # --- Run Metanome and time it ---
    start_time = time.time()
    command = [
        java_exe, java_memory, "-jar", metanome_jar,
        "--input-file", dataset_path,
        "--output-file", result_file
    ]

    try:
        subprocess.run(command, check=True)
        duration_ms = int((time.time() - start_time) * 1000)  # milliseconds

        # ✅ Append to performance log CSV
        with open(performance_log_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # Write header if file is empty
                writer.writerow(["Timestamp", "Dataset", "Size_MB", "Columns", "Metanome_Time_ms", "Output"])
            writer.writerow([
                timestamp(), dataset_path, f"{file_size_mb:.2f}", num_columns, duration_ms, result_file
            ])

    except subprocess.CalledProcessError as e:
        print(f"[{timestamp()}] ❌ Error running Metanome on {dataset_path}: {e}")


def train_random_forest_models():
    """Triggers the model training script for Random Forest."""
    print(f"Training multiple Random Forest models...")
    try:
        subprocess.run(["python3", "-m", "RandomForest.MultipleRandomForestTraining"], check=True)
        print(f"Model training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during model training: {e}")

def get_all_csv_files(base_dir):
    return glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)# ---- Main Pipeline ---- #
def main():
    print(f"[{timestamp()}] --- Starting full training dataset processing and model training pipeline ---")

    for data_type in data_types:
        data_dir = os.path.join(base_dir, data_type)
        result_dir = os.path.join(data_dir, "metanomeResults")

        os.makedirs(result_dir, exist_ok=True)
        clean_all_csv_files(data_dir)
        for csv_file in get_all_csv_files(data_dir):
            run_metanome_if_needed(csv_file, result_dir)

    train_random_forest_models()
    print(f"[{timestamp()}] --- Pipeline completed ---")
if __name__ == "__main__":
    main()