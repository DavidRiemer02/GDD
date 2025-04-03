import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("performance/training_performance_log.csv")

# Clean up Sampling column
df["Sampling"] = df["Sampling"].str.strip()

# Split by Sampling value
df_true = df[df["Sampling"] == "True"].copy()
df_false = df[df["Sampling"] == "False"].copy()

# Merge on Model_Name
merged_df = pd.merge(df_true, df_false, on="Model_Name", suffixes=("_with_sampling", "_without_sampling"))

# Compute training time difference
merged_df["Train_Time_Diff_ms"] = merged_df["Train_Time_ms_with_sampling"] - merged_df["Train_Time_ms_without_sampling"]

# Plot difference
plt.figure(figsize=(12, 6))
x = np.arange(len(merged_df["Model_Name"]))
plt.bar(x, merged_df["Train_Time_Diff_ms"], width=0.6, color="lightblue")
plt.xticks(ticks=x, labels=merged_df["Model_Name"], rotation=45, ha="right")
plt.axhline(0, color='gray', linewidth=0.8)
plt.ylabel("Training Time Difference (ms)")
plt.title("Difference in Training Time (With Sampling - Without Sampling)")
plt.tight_layout()
plt.savefig("train_time_difference.png", dpi=300)
plt.show()



color_map = {"True": "#1f77b4", "False": "#ff7f0e"}  # Blue for True, Orange for False
# Recreate the 3D scatter plot using solid circular markers ('o') for better visibility
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each group with distinct color and circular markers
for sampling_value in ["True", "False"]:
    subset = df[df["Sampling"] == sampling_value]
    ax.scatter(
        subset["Sample_Size"],
        subset["n_Estimators"],
        subset["Train_Time_ms"],
        label=f"Sampling: {sampling_value}",
        color=color_map[sampling_value],
        s=80,
        edgecolors='k',  # black border
        alpha=0.9,
        marker='o'       # solid circle marker
    )

# Labels and title
ax.set_xlabel("Sample Size")
ax.set_ylabel("n_Estimators")
ax.set_zlabel("Train Time (ms)")
ax.set_title("3D Visualization of Training Time (Grouped by Sampling)")
ax.legend()

plt.tight_layout()
plt.show()