
import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from Umap.UMAPColumnEmbedder import UMAPColumnEmbedder

# === CONFIG ===
real_data_folder = "TrainingData/realData"
fake_data_folder = "TrainingData/fakeData"
test_real_data_folder = "TestData/realData"
test_fake_data_folder = "TestData/fakeData"

# === Initialize Embedder ===
embedder = UMAPColumnEmbedder()

# === Helper: Recursively find all CSVs ===
def get_all_csv_files(folder):
    csv_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))
    return csv_files

# === Feature Extraction ===
def extract_umap_embeddings(csv_paths, label, embedder):
    rows = []
    for csv_path in csv_paths:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            umap_vector = embedder.transform_and_aggregate(numeric_df)
            row = {
                'umap_emb_0': umap_vector[0],
                'umap_emb_1': umap_vector[1],
                'umap_emb_2': umap_vector[2],
                'dataset': dataset_name,
                'label': label
            }
            rows.append(row)
        except Exception as e:
            print(f"❌ Failed to process {dataset_name}: {e}")
    return pd.DataFrame(rows) if rows else None

# === Load All Data ===
real_csvs = get_all_csv_files(real_data_folder)
fake_csvs = get_all_csv_files(fake_data_folder)
test_real_csvs = get_all_csv_files(test_real_data_folder)
test_fake_csvs = get_all_csv_files(test_fake_data_folder)

real_features_df = extract_umap_embeddings(real_csvs, 1, embedder)
fake_features_df = extract_umap_embeddings(fake_csvs, 0, embedder)
test_real_features_df = extract_umap_embeddings(test_real_csvs, 1, embedder)
test_fake_features_df = extract_umap_embeddings(test_fake_csvs, 0, embedder)

# === Combine ===
all_dfs = [df for df in [real_features_df, fake_features_df, test_real_features_df, test_fake_features_df] if df is not None]
combined_df = pd.concat(all_dfs, ignore_index=True)
print("Total samples:", len(combined_df))

# === Plot 3D UMAP Space ===
points = combined_df[['umap_emb_0', 'umap_emb_1', 'umap_emb_2']].values
labels = combined_df['label'].values
names = combined_df['dataset'].values

centroid = points.mean(axis=0)
distances = np.linalg.norm(points - centroid, axis=1)
threshold = np.percentile(distances, 95)
outlier_indices = np.where(distances > threshold)[0]

color_map = {1: 'lightblue', 0: 'pink'}
colors = [color_map[l] for l in labels]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=50, alpha=0.7)

# Optional: annotate outliers
# for idx in outlier_indices:
#     row = combined_df.iloc[idx]
#     ax.text(row['umap_emb_0'], row['umap_emb_1'], row['umap_emb_2'], row['dataset'], fontsize=6, color='black', alpha=0.8)

ax.set_title('3D UMAP Projection: Per-Column Embeddings', fontsize=18)
ax.set_xlabel('UMAP Dim 1')
ax.set_ylabel('UMAP Dim 2')
ax.set_zlabel('UMAP Dim 3')
legend_elements = [
    Patch(facecolor='lightblue', label='Real'),
    Patch(facecolor='pink', label='Fake')
]
ax.legend(handles=legend_elements, title='Label')
plt.tight_layout()
plt.show()
