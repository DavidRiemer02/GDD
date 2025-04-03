import os
import sys
import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import time
import csv
import pandas as pd
from datetime import datetime
from RandomForest.Utils.readlargeData import read_large_data
from RandomForest.MultipleRandomForestTraining import GeneratedDatasetDetector
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "RandomForest")))

# ---- Configuration ---- #
import json

# Load config from JSON
with open("config.json", "r") as f:
    config = json.load(f)

# Access variables
java_exe = config["java_exe"]
test_base_dir_real = config["test_base_dir"]
metanome_jar = config["metanome_jar"]
java_memory = config["java_memory"]

# Results directory
test_result_dir_real = os.path.join(test_base_dir_real, "metanomeResults")


# ---- Helper Functions ---- #
def clean_csv_in_place(file_path):
    """Cleans a CSV file in place by removing problematic quotes."""
    temp_output = file_path + ".tmp"
    read_large_data(file_path, temp_output)  # Call with both input and output paths
    os.replace(temp_output, file_path)  # Overwrite original file

def clean_all_csv_files(directory):
    """Recursively cleans all CSV files in a directory and its subdirectories."""
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    
    for csv_file in csv_files:
        clean_csv_in_place(csv_file)


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
        print(f"[{timestamp()}] Metanome result already exists: {result_file}, skipping.")
        return

    try:
        file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        df = pd.read_csv(dataset_path, nrows=5)
        num_columns = df.shape[1]
    except Exception as e:
        print(f"[{timestamp()}] Failed to read {dataset_path} for metadata: {e}")
        return

    # Run Metanome and log time
    print(f"[{timestamp()}] Running Metanome on: {dataset_path}")
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

        print(f"[{timestamp()}] Metanome completed in {duration_ms} ms")

    except subprocess.CalledProcessError as e:
        print(f"[{timestamp()}] Error running Metanome on {dataset_path}: {e}")


def get_all_csv_files(base_dir):
    return glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

def write_run_metadata(classification_results):
    summary_dir = "UserData/results"
    os.makedirs(summary_dir, exist_ok=True)
    summary_file_path = os.path.join(summary_dir, "run_summary.txt")

    summary_lines = []
    summary_lines.append(f"Run Timestamp: {timestamp()}")
    summary_lines.append("Run Configuration:")
    summary_lines.append(f"  Java Executable: {java_exe}")
    summary_lines.append(f"  Metanome JAR: {metanome_jar}")
    summary_lines.append(f"  Java Memory: {java_memory}")
    summary_lines.append(f"  Base Dataset Directory: {test_base_dir_real}")
    summary_lines.append("")

    summary_lines.append("Datasets Processed:")

    dataset_count = 0
    for csv_file in get_all_csv_files(test_base_dir_real):
        dataset_count += 1
        rel_path = os.path.relpath(csv_file, start=test_base_dir_real)
        try:
            file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
            df = pd.read_csv(csv_file, nrows=5)
            num_columns = df.shape[1]
            summary_lines.append(f"  - {rel_path} | {file_size_mb:.2f} MB | {num_columns} columns")
        except Exception as e:
            summary_lines.append(f"  - {rel_path} | ERROR: {str(e)}")

    summary_lines.append("")
    summary_lines.append(f"Total datasets classified: {dataset_count}")

    summary_lines.append("")
    summary_lines.append("Classification Results:")
    for file_path, label in classification_results:
        rel_path = os.path.relpath(file_path, start=test_base_dir_real)
        summary_lines.append(f"  - {rel_path}: {label}")

    # Write summary to file
    with open(summary_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"[{timestamp()}] Summary written to {summary_file_path}")



def test_pipeline():
    print("-----Starting full classification pipeline with metrics-----")

    for base_dir, result_dir in [(test_base_dir_real, test_result_dir_real)]:
        os.makedirs(result_dir, exist_ok=True)
        clean_all_csv_files(base_dir)
        for csv_file in get_all_csv_files(base_dir):
            run_metanome_if_needed(csv_file, result_dir)

    detector = GeneratedDatasetDetector()
    classification_results = detector.classify_new_datasets(test_base_dir_real)
    write_run_metadata(classification_results)
    #Delete Directory UserData/fakeData
    #Delete MetanomeResults Directory if exists
    if os.path.exists("UserData/metanomeResults"):
        os.rmdir("UserData/metanomeResults")

    print("-----Classification completed-----")


    
if __name__ == "__main__":
    test_pipeline()