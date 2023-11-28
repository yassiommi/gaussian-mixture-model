from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from gmm import *

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# Fit Gaussian Mixture Model
gmm = GaussianMixtureModel(n_components=4)
gmm.fit(X)

# Predict clusters
predicted_labels = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis', marker='o', edgecolor='black')
plt.title('Gaussian Mixture Model')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()
