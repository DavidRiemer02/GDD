import os
import subprocess
import glob
from RandomForest.Utils.cleanCSV import clean_csv_quotes
import sys


# ---- Configuration ---- #
java_exe = "C:\\Users\\David\\.jdks\\openjdk-18.0.2.1\\bin\\java"  # Full path to Java
base_dir = "TrainingData"
data_types = ["fakeData", "realData"]  # Subdirectories in TrainingData
metanome_jar = "generatedDatasetDetector.jar"  # Update this path
java_memory = "-Xmx8G"  # Adjust if needed
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "RandomForest")))

# ---- Helper Functions ---- #
def clean_csv_in_place(file_path):
    """Cleans a CSV file in place by removing problematic quotes."""
    temp_output = file_path + ".tmp"
    clean_csv_quotes(file_path, temp_output)  # Call with both input and output paths
    os.replace(temp_output, file_path)  # Overwrite original file

def clean_all_csv_files(directory):
    """Recursively cleans all CSV files in a directory and its subdirectories."""
    print(f"🔍 Cleaning CSV files in {directory} ...")
    csv_files = glob.glob(os.path.join(directory, "**", "*.csv"), recursive=True)
    
    for csv_file in csv_files:
        print(f"📂 Cleaning {csv_file} ...")
        clean_csv_in_place(csv_file)


def run_metanome_if_needed(dataset_path, result_dir):
    """Runs Metanome only if results do not already exist in the correct subfolder under metanomeResults."""
    base_name = os.path.basename(dataset_path).replace(".csv", "")

    # Find subfolder relative to base directory (e.g., "BenfordZipsDatasets")
    relative_path = os.path.relpath(os.path.dirname(dataset_path), start=os.path.dirname(result_dir))

    # Define correct result directory inside `metanomeResults/{subfolder}`
    metanome_result_folder = os.path.join(result_dir, relative_path)
    os.makedirs(metanome_result_folder, exist_ok=True)

    # Define correct result file path
    result_file = os.path.join(metanome_result_folder, f"{base_name}_Results.json")

    # Skip if JSON already exists
    if os.path.exists(result_file):
        print(f"✅ Metanome result already exists at {result_file}, skipping.")
        return

    # Run Metanome if JSON does not exist
    print(f"🚀 Running Metanome on {dataset_path} ...")
    command = [
        java_exe, java_memory, "-jar", metanome_jar,
        "--input-file", dataset_path,
        "--output-file", result_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"✅ Metanome finished for {dataset_path}, output saved: {result_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Metanome on {dataset_path}: {e}")


def train_random_forest_models():
    """Triggers the model training script for Random Forest."""
    print(f"🚀 Training multiple Random Forest models...")
    try:
        subprocess.run(["python3", "-m", "RandomForest.MultipleRandomForestTraining"], check=True)
        print(f"✅ Model training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during model training: {e}")


# ---- Main Pipeline ---- #
def main():
    """Main pipeline that processes datasets, extracts dependencies, and trains models."""
    print("🚀 Starting full dataset processing and model training pipeline ...")
    
    for data_type in data_types:
        data_dir = os.path.join(base_dir, data_type)
        result_dir = os.path.join(data_dir, "metanomeResults")

        # Step 1: Clean CSV files recursively
        clean_all_csv_files(data_dir)

        # Step 2: Run Metanome on each CSV file if results are missing
        csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        
        for csv_file in csv_files:
            run_metanome_if_needed(csv_file, result_dir)

    # Step 3: Train models
    train_random_forest_models()

    print("✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()