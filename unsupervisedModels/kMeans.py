from unsupervisedModels.Unsupervised import UnsupervisedModel
import numpy as np



class KMeansModel(UnsupervisedModel):
    def __init__(self, data, kmeans_model):
        """
        Initialize the KMeansModel class.

        Args:
            data (dict): Dictionary containing datasets (e.g., X_train, y_train, etc.).
            kmeans_model: Instance of KMeans clustering model.
        """
        super().__init__(data)
        self.kmeans_model = kmeans_model

    def train_and_evaluate(self, use_pca=False, n_components=2):
        """
        Train and evaluate the K-Means model.

        Args:
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            n_components (int): Number of principal components (if PCA is used).
        """
        # Load data
        X = self.data.get("X_train")
        y_true = self.data.get("y_train")

        if use_pca:
            X, _ = self.apply_pca(X, n_components=n_components)

        # Fit KMeans model
        self.kmeans_model.fit(X)
        y_pred = self.kmeans_model.labels_

        # Evaluate clustering performance
        results = self.evaluate_clustering(y_true, y_pred)

        # Plot clusters
        if use_pca:
            self.plot_clusters(X, y_pred, title="Clusters after PCA")
        else:
            self.plot_clusters(X, y_pred, title="Clusters without PCA")

        return results

# Example Usage
from sklearn.cluster import KMeans

