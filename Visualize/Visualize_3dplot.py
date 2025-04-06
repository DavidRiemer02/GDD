import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RandomForest.MultipleRandomForestTraining import GeneratedDatasetDetector
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from matplotlib.patches import Patch
from Umap.UMAPColumnEmbedder import UMAPColumnEmbedder


# Initialize the detector
detector = GeneratedDatasetDetector()
embedder = UMAPColumnEmbedder()


real_data_folder = "TrainingData/realData"
fake_data_folder = "TrainingData/fakeData"
real_metanome_folder = os.path.join(real_data_folder, "metanomeResults")
fake_metanome_folder = os.path.join(fake_data_folder, "metanomeResults")

def get_all_csv_files(folder):
    """Recursively collects all .csv files within a folder and its subfolders."""
    csv_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                full_path = os.path.join(root, f)
                csv_files.append(full_path)
    return csv_files

def extract_features(csv_paths, metanome_base_folder, label):
    """Extracts features for datasets with a central Metanome JSON folder."""
    feature_rows = []

    for csv_path in csv_paths:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        json_path = detector.find_metanome_json(dataset_name, metanome_base_folder)

        if not json_path:
            print(f"⚠️ No Metanome JSON found for {dataset_name}. Skipping.")
            continue

        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
            features = detector.extract_combined_features(df, json_path)
            features['dataset'] = dataset_name
            features['label'] = label
            feature_rows.append(features)
        except Exception as e:
            print(f"❌ Failed to process {dataset_name}: {e}")
            continue

    return pd.concat(feature_rows, ignore_index=True) if feature_rows else None

# Gather .csv files
real_csvs = get_all_csv_files(real_data_folder)
fake_csvs = get_all_csv_files(fake_data_folder)

# Extract features
real_features_df = extract_features(real_csvs, real_metanome_folder, label=1)
fake_features_df = extract_features(fake_csvs, fake_metanome_folder, label=0)

# Combine and save
if real_features_df is not None and fake_features_df is not None:
    combined_df = pd.concat([real_features_df, fake_features_df], ignore_index=True)
    print("Combined shape:", combined_df.shape)
else:
    print("Could not process all datasets.")

# === Load Test Data (real and fake) ===
test_real_data_folder = "TestData/realData"
test_fake_data_folder = "TestData/fakeData"
test_real_metanome_folder = os.path.join(test_real_data_folder, "metanomeResults")
test_fake_metanome_folder = os.path.join(test_fake_data_folder, "metanomeResults")

# Gather .csv files for test data
test_real_csvs = get_all_csv_files(test_real_data_folder)
test_fake_csvs = get_all_csv_files(test_fake_data_folder)

# Extract features
test_real_features_df = extract_features(test_real_csvs, test_real_metanome_folder, label=1)
test_fake_features_df = extract_features(test_fake_csvs, test_fake_metanome_folder, label=0)

# Combine all four datasets
all_dfs = [df for df in [real_features_df, fake_features_df, test_real_features_df, test_fake_features_df] if df is not None]
combined_df = pd.concat(all_dfs, ignore_index=True)
print("Combined shape including test data:", combined_df.shape)


reducer = umap.UMAP(n_components=3)
#drop name column and label column
labels = combined_df['label']
names = combined_df['dataset']
combined_df_droped = combined_df.drop(columns=['dataset', 'label'])
scaled_data = StandardScaler().fit_transform(combined_df_droped)
embedding = reducer.fit_transform(scaled_data)
embedding_df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2', 'Dim3'])
embedding_df['label'] = labels.values
embedding_df['dataset'] = names.values
points = embedding_df[['Dim1', 'Dim2', 'Dim3']].values

centroid = points.mean(axis=0)
distances = np.linalg.norm(points - centroid, axis=1)

# Identify outliers: top 5% by distance
threshold = np.percentile(distances, 95)
outlier_indices = np.where(distances > threshold)[0]
# Map labels to integers for color palette
colors = [sns.color_palette()[l] for l in embedding_df['label']]

color_map = {
    1: 'lightblue',  # real
    0: 'pink'        # fake
}
# Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = [color_map[l] for l in embedding_df['label']]

# Create the scatter plot in 3D
ax.scatter(
    embedding_df['Dim1'],
    embedding_df['Dim2'],
    embedding_df['Dim3'],
    c=colors,
    s=50,
    alpha=0.7
)

# ✅ Keep grid lines, but hide tick labels and ticks
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.tick_params(axis='both', which='both', length=0)  # hide ticks

# ✅ Keep pane lines (the cube grid)
ax.xaxis._axinfo["grid"]["linewidth"] = 0.5
ax.yaxis._axinfo["grid"]["linewidth"] = 0.5
ax.zaxis._axinfo["grid"]["linewidth"] = 0.5

# Title and legend
ax.set_title('3D UMAP Projection: Real vs Fake Data', fontsize=20)
legend_elements = [
    Patch(facecolor='lightblue', label='Real'),
    Patch(facecolor='pink', label='Fake')
]
ax.legend(handles=legend_elements, title="Label")

plt.show()