import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def collect_json_data(root_folder):
    data = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith('.json'):
                full_path = os.path.join(dirpath, file)
                rel_path = os.path.relpath(full_path, root_folder)
                with open(full_path, 'r') as f:
                    try:
                        data[rel_path] = json.load(f)
                    except Exception as e:
                        print(f"Failed to load {rel_path}: {e}")
    return data

# Load both folders
folder1 = "sampling/TrainingData"
folder2 = "nosampling/TrainingData"
data1 = collect_json_data(folder1)
data2 = collect_json_data(folder2)

# Compare JSON stats and compute differences
diff_rows = []
metrics = ["FDs_count", "UCCs_count", "INDs_count", "Max_FD_Length"]

for file in set(data1.keys()) & set(data2.keys()):
    d1, d2 = data1[file], data2[file]
    diff = {metric + "_diff": d2.get(metric, 0) - d1.get(metric, 0) for metric in metrics}
    diff["File"] = file
    diff_rows.append(diff)

# Create DataFrame
df_diff = pd.DataFrame(diff_rows)

# --- Average Difference Bar Plot ---
avg_diff = df_diff.drop(columns=["File"]).mean().rename(lambda x: x.replace("_diff", ""))

plt.figure(figsize=(8, 5))
sns.barplot(x=avg_diff.index, y=avg_diff.values, color="lightblue")
plt.title("Average Difference per Metric (No Sampling - Sampling)")
plt.ylabel("Average Difference")
plt.xlabel("Metric")
plt.grid(axis='y')
plt.tight_layout()
plt.show()