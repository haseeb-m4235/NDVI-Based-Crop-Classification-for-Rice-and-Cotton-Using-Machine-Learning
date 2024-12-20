from unsupervisedModels.Unsupervised import UnsupervisedModel

class GaussianMixtureModel(UnsupervisedModel):
    def __init__(self, data, gmm_model):
        """
        Initialize the GaussianMixtureModel class.

        Args:
            data (dict): Dictionary containing datasets (e.g., X_train, y_train, etc.).
            gmm_model: Instance of Gaussian Mixture Model clustering.
        """
        super().__init__(data)
        self.gmm_model = gmm_model

    def train_and_evaluate(self, use_pca=False, n_components=2):
        """
        Train and evaluate the Gaussian Mixture Model.

        Args:
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            n_components (int): Number of principal components (if PCA is used).
        """
        # Load data
        X = self.data.get("X_train")
        y_true = self.data.get("y_train")

        if use_pca:
            X, _ = self.apply_pca(X, n_components=n_components)

        # Fit Gaussian Mixture Model
        self.gmm_model.fit(X)
        y_pred = self.gmm_model.predict(X)

        # Evaluate clustering performance
        results = self.evaluate_clustering(y_true, y_pred)

        # Plot clusters
        if use_pca:
            self.plot_clusters(X, y_pred, title="Gaussian Mixture Clusters after PCA")
        else:
            self.plot_clusters(X, y_pred, title="Gaussian Mixture Clusters without PCA")

        return results
