import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

def visualize_first_decision_tree(model_path):
    """Visualizes the first decision tree from a trained Random Forest model."""
    
    # Check if the model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    # Load the trained Random Forest model
    try:
        rf_classifier = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Ensure the model has estimators
    if not hasattr(rf_classifier, "estimators_") or len(rf_classifier.estimators_) == 0:
        print("❌ Error: The Random Forest model does not contain any decision trees.")
        return
    
    # Define feature names (Ensure these match the features used during training)
    feature_names = [
        'Mean of all numerical values', 'Standard Deviation of numerical values',
        'Minimum value in numerical columns', 'Maximum value in numerical columns',
        'Skewness of numerical values', 'Kurtosis of numerical values',
        'Deviation from Benford\'s Law using MAE',
        'Number of categorical columns in the dataset',
        'Average number of unique values per categorical column',
        'Most frequent category\'s percentage in each categorical column',
        'Entropy of categorical distributions',
        'Correlation with expected Zipfian distribution',
        'Ratio of numerical to categorical columns',
        'Functional Dependencies Ratio', 'Unique Column Combinations Ratio',
        'Inclusion Dependencies Ratio', 'Max FD Length Normalized',
        'Functional Dependencies Count', 'Unique Column Combinations Count',
        'Inclusion Dependencies Count', 'Max FD Length'
    ]
    
    # Ensure feature name count matches model input dimensions
    if len(feature_names) != len(rf_classifier.feature_importances_):
        print(f"Warning: Feature name count ({len(feature_names)}) does not match model input ({len(rf_classifier.feature_importances_)})")
        feature_names = [f"Feature {i}" for i in range(len(rf_classifier.feature_importances_))]  # Assign default names
    
    # Select the first tree in the ensemble
    first_tree = rf_classifier.estimators_[0]

    # Plot the first decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(first_tree, filled=True, feature_names=feature_names, rounded=True, fontsize=8)
    plt.title("First Decision Tree from Random Forest Model")
    plt.show()

# Example usage:
model_path = "models/randomForest/random_forest_s500_n100_d10.pkl"
visualize_first_decision_tree(model_path)
