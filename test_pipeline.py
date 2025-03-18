import os
import subprocess
import glob
import argparse
from RandomForest.Utils.cleanCSV import clean_csv_quotes
from RandomForest.MultipleRandomForestTraining import GeneratedDatasetDetector

# ---- Configuration ---- #
java_exe = "C:\\Users\\David\\.jdks\\openjdk-18.0.2.1\\bin\\java"  # Full path to Java
test_base_dir = "TestData/realData"  # Test dataset directory (change for realData if needed)
metanome_jar = "generatedDatasetDetector.jar"  # JAR file path
java_memory = "-Xmx8G"  # Adjust memory as needed

# Results directory
test_result_dir = os.path.join(test_base_dir, "metanomeResults")


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


def run_metanome_if_needed(dataset_path, result_dir):
    """Runs Metanome only if results do not already exist in the correct subfolder under metanomeResults."""
    base_name = os.path.basename(dataset_path).replace(".csv", "")

    # Find subfolder relative to test_base_dir (e.g., "BenfordZipsDatasets")
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


# ---- Test Pipeline ---- #
def test_pipeline():
    """Main pipeline for processing test datasets, extracting dependencies, and classifying."""
    print("🚀 Starting test dataset processing ...")

    # Step 1: Ensure metanomeResults directory exists
    os.makedirs(test_result_dir, exist_ok=True)

    # Step 2: Clean CSV files recursively
    clean_all_csv_files(test_base_dir)

    # Step 3: Run Metanome on each CSV file if results are missing
    csv_files = glob.glob(os.path.join(test_base_dir, "**", "*.csv"), recursive=True)
    
    for csv_file in csv_files:
        run_metanome_if_needed(csv_file, test_result_dir)

    # Step 4: Classify datasets using GeneratedDatasetDetector
    detector = GeneratedDatasetDetector()
    detector.classify_new_datasets(test_base_dir)

    print("✅ Test pipeline completed successfully.")


if __name__ == "__main__":
    test_pipeline()
