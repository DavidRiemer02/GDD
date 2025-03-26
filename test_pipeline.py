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
        print(f"Metanome result already exists at {result_file}, skipping.")
        return

    # Run Metanome if JSON does not exist
    print(f"Running Metanome on {dataset_path} ...")
    command = [
        java_exe, java_memory, "-jar", metanome_jar,
        "--input-file", dataset_path,
        "--output-file", result_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Metanome finished for {dataset_path}, output saved: {result_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Metanome on {dataset_path}: {e}")
def get_all_csv_files(base_dir):
    return glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)

# ---- Test Pipeline ---- #
from sklearn.metrics import accuracy_score

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