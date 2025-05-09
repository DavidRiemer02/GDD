import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
df = pd.read_csv("performance/classification_log.csv")
df.columns = df.columns.str.strip()

# Select only the models you want to plot
selected_models = [
    "random_forest_grid_search.pkl",
    "random_forest_s500_n1_d20.pkl",
    "random_forest_s500_n100_d10.pkl"
]

df_filtered = df[df["Model"].isin(selected_models)]

# Define custom colors
color_map = {
    "random_forest_grid_search.pkl": "pink",
    "random_forest_s500_n1_d20.pkl": "lightblue",
    "random_forest_s500_n100_d10.pkl": "purple"
}

# Create plot
plt.figure(figsize=(12, 6))

# Plot each model as scatter points with reduced alpha
for model in selected_models:
    subset = df_filtered[df_filtered["Model"] == model]
    plt.scatter(
        subset["Size_MB"], 
        subset["Classification_Time_ms"]*10, 
        label=model, 
        color=color_map[model], 
        s=80,
        alpha=0.6  # Set transparency (1.0 = opaque, 0.0 = invisible)
    )

# Customize plot
plt.title("Classification Time vs Dataset Size for Selected Models (Sample)")
plt.xlabel("Size (MB)")
plt.ylabel("Classification Time (ms)")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.show()
# Calculate average classification time per model
avg_time_per_model = df_filtered.groupby("Model")["Classification_Time_ms"].mean().reset_index()
avg_time_per_model.columns = ["Model", "Average_Classification_Time_ms"]
avg_time_per_model = avg_time_per_model.sort_values(by="Average_Classification_Time_ms", ascending=False)

# Display result
avg_time_per_model

#Plot avg_time_per_model
plt.figure(figsize=(10, 6))

sns.barplot(x="Model", y="Average_Classification_Time_ms", 
            data=avg_time_per_model, 
            palette=[color_map.get(model, "gray") for model in avg_time_per_model["Model"]]
)
plt.title("Average Classification Time per Model")
plt.xlabel("Model")
plt.ylabel("Average Classification Time (ms)")
plt.xticks(rotation=45)
plt.tight_layout()
#Label the bars with the exact values
for i, v in enumerate(avg_time_per_model["Average_Classification_Time_ms"]):
    plt.text(i, v + 10, f"{v:.1f}", ha="center")
plt.show()