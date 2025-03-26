import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load performance log
df = pd.read_csv("performance/metanome_performance_log.csv")

# 3D Scatter Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(df["Size_MB"], df["Columns"], df["Metanome_Time_ms"],
                c=df["Metanome_Time_ms"], cmap="plasma", s=80)

ax.set_xlabel("Dataset Size (MB)")
ax.set_ylabel("Number of Columns")
ax.set_zlabel("Metanome Time (ms)")
plt.title("Metanome Runtime vs Dataset Size and Number of Columns")

fig.colorbar(sc, label="Runtime (ms)")
plt.tight_layout()
plt.show()


import seaborn as sns

sns.lmplot(data=df, x="Size_MB", y="Metanome_Time_ms", hue="Columns",
           palette="coolwarm", height=6, aspect=1.5, scatter_kws={"s": 60})
plt.title("Metanome Time vs Dataset Size, colored by Columns")
plt.tight_layout()
plt.show()