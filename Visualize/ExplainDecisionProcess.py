import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from RandomForest.MultipleRandomForestTraining import *

def explain_decision_process(model_path, dataset_path):
    """
    Explains how a trained Random Forest model classifies a single dataset.
    
    Args:
        model_path (str): Path to the trained Random Forest model (.pkl file).
        dataset_path (str): Path to the dataset (.csv file).
    """
    # Load the trained model
    rf_classifier = joblib.load(model_path)

    # Initialize the detector class
    detector = GeneratedDatasetDetector()

    # Extract dataset name (without extension) and find Metanome JSON path
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    meta_folder = os.path.join(os.path.dirname(dataset_path), "metanomeResults")
    json_path = detector.find_metanome_json(dataset_name, meta_folder)

    if not json_path:
        print(f"Warning: No Metanome JSON found for {dataset_name}. Skipping feature extraction.")
        return

    # Load the dataset
    df = pd.read_csv(dataset_path)
    if df.empty:
        print("❌ Error: The dataset is empty.")
        return
    
    # Extract features using the detector
    feature_df = detector.extract_combined_features(df, json_path)

    # Convert DataFrame to NumPy array for model input
    X = feature_df.values  

    # Get predictions from all trees in the ensemble
    tree_predictions = np.array([tree.predict(X) for tree in rf_classifier.estimators_])
    
    # Calculate the majority vote from all trees
    final_prediction = np.round(tree_predictions.mean(axis=0)).astype(int)

    print("\n🔍 **Random Forest Decision Process**")
    print(f"Total Trees in Forest: {len(rf_classifier.estimators_)}")
    print(f"Tree Predictions: {tree_predictions[:, 0]}")  # Show first dataset's predictions
    print(f"Final Majority Vote Prediction: {'Real' if final_prediction[0] == 1 else 'Fake'}")

    # Get feature names from DataFrame
    feature_names = feature_df.columns.tolist()

    # ---- Visualize First Decision Tree ----
    plt.figure(figsize=(20, 10))
    plot_tree(rf_classifier.estimators_[0], filled=True, feature_names=feature_names, rounded=True, fontsize=8)
    plt.title("First Decision Tree from Random Forest Model (Decision Path)")
    plt.show()

# Example usage
model_path = "models/randomForest/random_forest_s5000_n1000_d50.pkl"
dataset_path = "TestData/realData/meets.csv"

explain_decision_process(model_path, dataset_path)
