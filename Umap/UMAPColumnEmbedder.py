import numpy as np
import sys
import joblib
from sklearn.preprocessing import StandardScaler
import umap
import os
from scipy.stats import skew, kurtosis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RandomForest.Utils.benford_analysis import benford_deviation as bd

class UMAPColumnEmbedder:
    def __init__(self, model_dir="models/umapRandomForest", n_components=3):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.umap_model_path = os.path.join(self.model_dir, "umap_model.pkl")
        self.umap_scaler_path = os.path.join(self.model_dir, "umap_scaler.pkl")
        self.n_components = n_components

    def extract_per_column_features(self, numeric_df):
        column_features = []
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) < 3:
                continue
            try:
                features = [
                    col_data.mean(),
                    col_data.std(),
                    col_data.min(),
                    col_data.max(),
                    skew(col_data),
                    kurtosis(col_data),
                    bd(col_data)
                ]
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                column_features.append(features)
            except Exception:
                continue  # Skip column if numerical stats fail
        return np.array(column_features)

    def fit_umap(self, list_of_column_feature_arrays):
        all_column_vectors = np.vstack(list_of_column_feature_arrays)
        self.scaler = StandardScaler().fit(all_column_vectors)
        X_scaled = self.scaler.transform(all_column_vectors)
        self.umap_model = umap.UMAP(n_components=self.n_components, random_state=42).fit(X_scaled)
        joblib.dump(self.umap_model, self.umap_model_path)
        joblib.dump(self.scaler, self.umap_scaler_path)

    def load_model(self):
        self.umap_model = joblib.load(self.umap_model_path)
        self.scaler = joblib.load(self.umap_scaler_path)

    def transform_and_aggregate(self, numeric_df):
        if not hasattr(self, "umap_model") or not hasattr(self, "scaler"):
            self.load_model()
        col_feat = self.extract_per_column_features(numeric_df)
        if col_feat.shape[0] == 0:
            return np.zeros(self.n_components)
        col_feat_scaled = self.scaler.transform(col_feat)
        embedded = self.umap_model.transform(col_feat_scaled)
        return embedded.mean(axis=0)
