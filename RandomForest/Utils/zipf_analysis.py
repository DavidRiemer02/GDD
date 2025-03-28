import numpy as np
import pandas as pd
from scipy.stats import linregress

def zipf_correlation(series):
    """Improved Zipf fit analysis: returns Pearson r and RMSE in log-log space."""
    counts = series.value_counts(normalize=True).sort_values(ascending=False).values
    counts = counts[counts > 0]  # avoid log(0)
    
    if len(counts) < 2:
        return {"r_value": np.nan, "rmse": np.nan}
    
    ranks = np.arange(1, len(counts) + 1)
    log_ranks = np.log(ranks)
    log_counts = np.log(counts)

    slope, intercept, r_value, _, _ = linregress(log_ranks, log_counts)
    residuals = log_counts - (slope * log_ranks + intercept)
    rmse = np.sqrt(np.mean(residuals**2))

    return r_value


# Load and process the dataset
df = pd.read_csv("TrainingData/fakeData/BenfordZipsDatasets/Perfect_Zipf_Dataset__10_Columns_.csv")
categorical_df = df.select_dtypes(include=['object', 'category'])

# Apply to the stacked series
result = zipf_correlation(categorical_df.stack())