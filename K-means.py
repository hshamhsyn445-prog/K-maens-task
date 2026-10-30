import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create the dataset
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [8, 8],
    [9, 8],
    [8, 9],
    [9, 10],
    [15, 15],
    [16, 15],
    [15, 16],
    [16, 17]
])

# Set the number of clusters
k = 3

# Create the K-Means model
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

# Train the model
kmeans.fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Get the cluster centers
centers = kmeans.cluster_centers_

# Print the results
print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Visualize the clusters
for i in range(k):
    plt.scatter(
        X[labels == i, 0],
        X[labels == i, 1],
        label=f"Cluster {i}"
    )

# Plot the cluster centers
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

# Add chart labels
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

# Display the chart
plt.show()
