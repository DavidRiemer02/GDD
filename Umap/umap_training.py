from UMAPColumnEmbedder_Advanced import UMAPColumnEmbedder
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

embedder = UMAPColumnEmbedder()

# Collect column-level features from all real and fake datasets
def get_column_feature_arrays(folder_paths):
    vectors = []
    for path in folder_paths:
        try:
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")
            numeric_df = df.select_dtypes(include=["int64", "float64"])
            if numeric_df.empty:
                continue
            features = embedder.extract_per_column_features(numeric_df)
            if features.size > 0:
                vectors.append(features)
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return vectors

# Get file paths
real_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk("TrainingData/realData") for f in filenames if f.endswith(".csv")]
fake_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk("TrainingData/fakeData") for f in filenames if f.endswith(".csv")]

# Fit UMAP model on column-wise statistics
all_vectors = get_column_feature_arrays(real_paths + fake_paths)
embedder.fit_umap(all_vectors)
print("UMAP embedder and scaler fitted and saved.")

def get_umap_embeddings(paths, label):
    rows = []
    for p in paths:
        try:
            df = pd.read_csv(p, on_bad_lines="skip", engine="python")
            numeric_df = df.select_dtypes(include=["int64", "float64"])
            if numeric_df.empty:
                continue
            emb = embedder.transform_and_aggregate(numeric_df)
            rows.append(np.append(emb, label))
        except Exception as e:
            print(f"❌ Failed to embed {p}: {e}")
    return rows

# Embed datasets
real_embs = get_umap_embeddings(real_paths, 1)
fake_embs = get_umap_embeddings(fake_paths, 0)

data = np.vstack(real_embs + fake_embs)
X, y = data[:, :-1], data[:, -1]

# Step 1: Embed TrainingData
train_real_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk("TrainingData/realData") for f in filenames if f.endswith(".csv")]
train_fake_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk("TrainingData/fakeData") for f in filenames if f.endswith(".csv")]

train_real_embs = get_umap_embeddings(train_real_paths, 1)
train_fake_embs = get_umap_embeddings(train_fake_paths, 0)

X_train = np.vstack(train_real_embs + train_fake_embs)
y_train = X_train[:, -1]
X_train = X_train[:, :-1]

# Step 2: Embed TestData
test_real_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk("TestData/realData") for f in filenames if f.endswith(".csv")]
test_fake_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk("TestData/fakeData") for f in filenames if f.endswith(".csv")]

test_real_embs = get_umap_embeddings(test_real_paths, 1)
test_fake_embs = get_umap_embeddings(test_fake_paths, 0)

X_test = np.vstack(test_real_embs + test_fake_embs)
y_test = X_test[:, -1]
X_test = X_test[:, :-1]

# Step 3: Train & Evaluate
clf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
os.makedirs("models/umapRandomForest", exist_ok=True)
joblib.dump(clf, "models/umapRandomForest/random_forest_umap.pkl")

print("\n🎯 Test Data Performance:")
print(classification_report(y_test, clf.predict(X_test), target_names=["Fake", "Real"]))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], X_test[:,2], c=y_test, cmap='coolwarm')
plt.title("UMAP-Embedded Test Datasets")
plt.show()
