import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os

# ✅ Load the trained Random Forest model
model_path = "models/randomForest/random_forest_s2000_n1000_d5.pkl"
rf_classifier = joblib.load(model_path)

# ✅ Define feature names based on the complete set of extracted features (Statistical + Metanome raw & ratio)
feature_names = [
    # Statistical Features
    'Mean of all numerical values',
    'Standard Deviation of numerical values',
    'Minimum value in numerical columns',
    'Maximum value in numerical columns',
    'Skewness of numerical values',
    'Kurtosis of numerical values',
    'Deviation from Benford\'s Law (MAE)',
    'Number of categorical columns',
    'Average number of unique values per categorical column',
    'Most frequent category\'s percentage in each categorical column',
    'Entropy of categorical distributions',
    'Zipf correlation of categorical distributions',
    
    # Metanome Dependency-Based Features (Normalized Ratios)
    'FDs ratio (FDs / Columns)',
    'UCCs ratio (UCCs / Columns)',
    'INDs ratio (INDs / Columns)',
    'Max FD length normalized (Max FD Length / Columns)',
    
    # Metanome Dependency-Based Features (Raw Counts)
    'FDs count (Functional Dependencies)',
    'UCCs count (Unique Column Combinations)',
    'INDs count (Inclusion Dependencies)',
    'Max FD length (Maximum FD Length)'
]

# ✅ Check consistency: the number of features should match the model input dimension
if len(feature_names) != len(rf_classifier.feature_importances_):
    raise ValueError(
        f"❌ Number of feature names ({len(feature_names)}) does not match model's expected input features ({len(rf_classifier.feature_importances_)}). "
        "Check feature extraction and training consistency!"
    )

# ✅ Extract feature importances from the trained model
feature_importances = rf_classifier.feature_importances_

# ✅ Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)  # For better horizontal bar visualization

# ✅ Plot Feature Importance
plt.figure(figsize=(14, 10))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='pink')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest Classifier (Statistical + Metanome Features)")
plt.tight_layout()
plt.show()
