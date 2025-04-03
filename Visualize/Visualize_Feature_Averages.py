import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# Add the root directory (where RandomForest lives) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
            try:
                df = pd.read_csv(dataset_path, on_bad_lines='warn')  # or 'skip'
            except Exception as e:
                print(f"⚠️ Skipping file {csv_file} due to read error: {e}")
                continue
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
    sns.barplot(x='Feature', y='Fake', data=avg_df, color="pink", label="Fake Data")

    plt.xticks(rotation=90)  # Rotate feature names for better readability
    plt.ylabel("Average Feature Value (Log Scale)", fontsize=16)
    plt.xlabel("Feature", fontsize=16)
    plt.title("Feature Averages for Real vs Fake Data (Log Scale)")
    plt.yscale("log")  # Apply log scale to y-axis
    plt.legend()
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    # Save the visualization
    output_path = "feature_averages_log.png"
    plt.savefig(output_path, dpi=300)
    print(f"📊 Saved feature visualization to: {output_path}")

    plt.show()
else:
    print("⚠️ Could not compute feature averages. Check dataset paths and formats.")


# Compute absolute difference between real and fake averages
avg_df['Difference'] = avg_df['Real'] - avg_df['Fake']

# Plot the difference
plt.figure(figsize=(14, 7))
sns.barplot(x='Feature', y='Difference', data=avg_df, palette="coolwarm")

plt.xticks(rotation=90)
plt.ylabel("Absolute Difference (Real - Fake)")
plt.xlabel("Feature")
plt.title("Absolute Difference: Real - Fake (Feature Averages)")
plt.axhline(0, color='black', linestyle='--')
plt.ylim(bottom=-6500000, top=1000000)
plt.tight_layout()

# Save the plot
diff_plot_path = "feature_averages_difference.png"
plt.savefig(diff_plot_path, dpi=300)
print(f"📊 Saved difference plot to: {diff_plot_path}")
plt.show()



import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Load the trained model
model_path = "models/randomForest/random_forest_s500_n1_d20.pkl"
rf_classifier = joblib.load(model_path)

# Updated list of feature names based on the user's implementation
feature_names_updated = [
    # Statistical Features
    'num_mean',
    'num_std',
    'num_min',
    'num_max',
    'num_skew',
    'num_kurtosis',
    'benford_mae',
    'num_categorical',
    'cat_unique_ratio',
    'cat_mode_freq',
    'cat_entropy',
    'zipf_corr',
    'num_vs_cat_ratio',
    
    # Metanome Dependency Features
    'fds_ratio',
    'uccs_ratio',
    'inds_ratio',
    'max_fd_length_norm',
    'fds_count',
    'uccs_count',
    'inds_count',
    'max_fd_length'
]

# Attempt to load the best model from grid search
model_path = "models/randomForest/random_forest_grid_search.pkl"
rf_classifier = joblib.load(model_path)

# Check alignment
assert len(feature_names_updated) == len(rf_classifier.feature_importances_), \
    f"Feature count mismatch: {len(feature_names_updated)} names vs {len(rf_classifier.feature_importances_)} importances."

# Plot the Top 10 importances
importance_df = pd.DataFrame({
    'Feature': feature_names_updated,
    'Importance': rf_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False).head(10)  # Select the top 10 features
# Select the top 10 features

# Plot
plt.figure(figsize=(16, 9))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightblue')
#increase size of the bar labels
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel("Importance Score", fontsize=20)
plt.ylabel("Feature", fontsize=20)
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
# Save the output
output_path = "feature_importance_grid_search.png"
plt.savefig(output_path, dpi=300)
output_path
