import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Confusion matrix data
conf_matrix = np.array([[17, 6],
                        [17, 60]])

# Labels for the axes
labels = ['Fake', 'Real']

lightpink_cmap = LinearSegmentedColormap.from_list("lightpink", ["mistyrose", "lightpink", "hotpink"])

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=lightpink_cmap,
            xticklabels=labels, yticklabels=labels, cbar=False
            ,annot_kws={"size": 21})

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (500 samples, 100 tree, 10 depth)')
plt.tight_layout()
plt.show()
