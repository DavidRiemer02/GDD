import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Define directories
real_data_directory = "TestData/realData/metanomeResults"
fake_data_directory = "TestData/fakeData/metanomeResults"

# Initialize dictionaries to store summed values and counts
real_data = defaultdict(list)
fake_data = defaultdict(list)

# Function to process JSON files
def process_json_files(directory, data_dict):
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                for key in ["FDs_count", "UCCs_count", "NumberOfColumns", "Max_FD_Length", "INDs_count"]:
                    if key in data:
                        data_dict[key].append(data[key])

# Process real and fake data directories
process_json_files(real_data_directory, real_data)
process_json_files(fake_data_directory, fake_data)

# Compute averages
real_avg = {col: np.mean(values) for col, values in real_data.items()}
fake_avg = {col: np.mean(values) for col, values in fake_data.items()}

# Create DataFrame for visualization
df_real = pd.DataFrame(list(real_avg.items()), columns=["Feature", "Real Avg"])
df_fake = pd.DataFrame(list(fake_avg.items()), columns=["Feature", "Fake Avg"])
df = pd.merge(df_real, df_fake, on="Feature", how="outer")

# Plot distributions
plt.figure(figsize=(12, 6))
x = np.arange(len(df["Feature"]))
width = 0.4

plt.bar(x - width/2, df["Real Avg"], width, label="Real", color="pink")
plt.bar(x + width/2, df["Fake Avg"], width, label="Fake", color="lightblue")

plt.xticks(ticks=x, labels=df["Feature"], rotation=45)
plt.xlabel("Features")
plt.ylabel("Average Count")
plt.title("Distribution of Dependencies per Feature (Real vs Fake)")
plt.legend()
plt.show()