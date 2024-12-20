from scipy.cluster.hierarchy import dendrogram, linkage
from unsupervisedModels.Unsupervised import UnsupervisedModel
import matplotlib.pyplot as plt
import numpy as np

class HierarchicalClusteringModel(UnsupervisedModel):
    def __init__(self, data):
        """
        Initialize the HierarchicalClusteringModel class.

        Args:
            data (dict): Dictionary containing datasets (e.g., X_train, y_train, etc.).
        """
        super().__init__(data)

    def train_and_evaluate(self, linkage_method="ward", use_pca=False, n_components=2):
        """
        Train and evaluate the Hierarchical Clustering model.

        Args:
            linkage_method (str): Linkage method for hierarchical clustering (e.g., 'ward', 'single', 'complete').
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            n_components (int): Number of principal components (if PCA is used).
        """
        # Load data
        X = self.data.get("X_train")
        y_true = self.data.get("y_train")

        if use_pca:
            X, _ = self.apply_pca(X, n_components=n_components)

        # Perform hierarchical clustering
        linkage_matrix = linkage(X, method=linkage_method)
        
        # Create dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.title("Dendrogram")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.show()

        # Assign clusters (use number of true labels as guidance)
        from scipy.cluster.hierarchy import fcluster
        y_pred = fcluster(linkage_matrix, t=len(np.unique(y_true)), criterion="maxclust")

        # Evaluate clustering performance
        results = self.evaluate_clustering(y_true, y_pred)

        # Plot clusters
        if use_pca:
            self.plot_clusters(X, y_pred, title="Hierarchical Clusters after PCA")
        else:
            self.plot_clusters(X, y_pred, title="Hierarchical Clusters without PCA")

        return results