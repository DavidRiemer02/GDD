import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained Decision Tree model
dt_classifier = joblib.load("models/decisionTree/decision_tree_multi_real.pkl")  # Update with your model path if needed

# Define feature names based on the extracted statistical features used during training
feature_names = [
    'num_mean', 'num_std', 'num_min', 'num_max', 'num_skew', 'num_kurtosis',
    'num_categorical', 'cat_unique_ratio', 'cat_mode_freq', 'cat_entropy'
]

# Extract feature importances from the trained model
feature_importances = dt_classifier.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Decision Tree Classifier")
plt.gca().invert_yaxis()  # Highest importance at the top
plt.show()
