import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from RandomForest.MultipleRandomForestTraining import GeneratedDatasetDetector

# Initialize the detector
detector = GeneratedDatasetDetector()

# Define data paths
real_data_folder = "TrainingData/realData"
fake_data_folder = "TrainingData/fakeData"

def compute_average_features(folder):
    """Extracts and averages features across all datasets in a given folder."""
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    feature_list = []
    
    for csv_file in csv_files:
        dataset_path = os.path.join(folder, csv_file)
        json_path = detector.find_metanome_json(os.path.splitext(csv_file)[0], os.path.join(folder, "metanomeResults"))

        if json_path:
            df = pd.read_csv(dataset_path)
            features = detector.extract_combined_features(df, json_path)
            feature_list.append(features)
    
    if not feature_list:
        print(f"⚠️ No valid datasets found in {folder}")
        return None
    
    # Compute mean feature values
    return pd.concat(feature_list).mean()

# Compute average feature values for real and fake data
real_means = compute_average_features(real_data_folder)
fake_means = compute_average_features(fake_data_folder)

if real_means is not None and fake_means is not None:
    # Prepare DataFrame for visualization
    avg_df = pd.DataFrame({'Feature': real_means.index, 'Real': real_means.values, 'Fake': fake_means.values})
    avg_df = avg_df.sort_values(by='Real', ascending=False)

    # Plot the feature averages with a log scale
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Feature', y='Real', data=avg_df, color="lightblue", label="Real Data")
    sns.barplot(x='Feature', y='Fake', data=avg_df, color="salmon", label="Fake Data")

    plt.xticks(rotation=90)  # Rotate feature names for better readability
    plt.ylabel("Average Feature Value (Log Scale)")
    plt.xlabel("Feature")
    plt.title("Feature Averages for Real vs Fake Data (Log Scale)")
    plt.yscale("log")  # Apply log scale to y-axis
    plt.legend()
    plt.tight_layout()

    # Save the visualization
    output_path = "feature_averages_log.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 Saved feature visualization to: {output_path}")

    plt.show()
else:
    print("⚠️ Could not compute feature averages. Check dataset paths and formats.")
