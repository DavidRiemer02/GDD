import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load performance logs
df1 = pd.read_csv("performance/metanome_performance_log_sampling.csv")
df2 = pd.read_csv("performance/metanome_performance_log.csv")

# Add run labels
df1['Run'] = 'With Sampling'
df2['Run'] = 'Without Sampling'

# Ensure consistent column naming for merging
df1 = df1.rename(columns={"Metanome_Time_ms": "Metanome_Time"})
df2 = df2.rename(columns={"Metanome_Time_ms": "Metanome_Time"})

# Combine both runs
df = pd.concat([df1, df2], ignore_index=True)

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each run separately to color by source
for run_label, color in zip(['With Sampling', 'Without Sampling'], ['lightblue', 'pink']):
    subset = df[df['Run'] == run_label]
    ax.scatter(
        subset['Size_MB'],
        subset['Columns'],
        subset['Metanome_Time'],
        label=run_label,
        c=color,
        s=80,
        edgecolor='k',
        alpha=0.8
    )

# Labels and legend
ax.set_xlabel("Dataset Size (MB)")
ax.set_ylabel("Number of Columns")
ax.set_zlabel("Metanome Time (ms)")
plt.title("Metanome Runtime Comparison (3D)")

ax.legend()
plt.tight_layout()
plt.show()
