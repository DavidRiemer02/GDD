import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("performance/training_performance_log.csv")

# Convert boolean to string for categorical plotting
df["Use_GridSearch"] = df["Use_GridSearch"].astype(str)

# Set Seaborn style
sns.set(style="whitegrid")

# Create the lineplot with custom colors
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="Sample_Size",
    y="Train_Time_ms",
    hue="Use_GridSearch",
    marker="o",
    markersize=10,
    palette=["pink", "lightblue"]
)

# Label and title
plt.title("Train Time vs Sample Size")
plt.ylabel("Train Time (ms)")
plt.xlabel("Sample Size")
plt.tight_layout()
plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df["Sample_Size"], df["n_Estimators"], df["Train_Time_ms"],
                c=df["Train_Time_ms"], cmap="viridis", s=80)
ax.set_xlabel("Sample Size")
ax.set_ylabel("n_Estimators")
ax.set_zlabel("Train Time (ms)")
plt.title("3D Visualization of Training Time")
fig.colorbar(sc, label='Train Time (ms)')
plt.show()