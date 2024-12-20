from unsupervisedModels.Unsupervised import UnsupervisedModel

class DBSCANModel(UnsupervisedModel):
    def __init__(self, data, dbscan_model):
        """
        Initialize the DBSCANModel class.

        Args:
            data (dict): Dictionary containing datasets (e.g., X_train, y_train, etc.).
            dbscan_model: Instance of DBSCAN clustering model.
        """
        super().__init__(data)
        self.dbscan_model = dbscan_model

    def train_and_evaluate(self, use_pca=False, n_components=2):
        """
        Train and evaluate the DBSCAN model.

        Args:
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            n_components (int): Number of principal components (if PCA is used).
        """
        # Load data
        X = self.data.get("X_train")
        y_true = self.data.get("y_train")

        if use_pca:
            X, _ = self.apply_pca(X, n_components=n_components)

        # Fit DBSCAN model
        y_pred = self.dbscan_model.fit_predict(X)
        

        # Evaluate clustering performance
        results = self.evaluate_clustering(y_true, y_pred)

        # Plot clusters
        if use_pca:
            self.plot_clusters(X, y_pred, title="DBSCAN Clusters after PCA")
        else:
            self.plot_clusters(X, y_pred, title="DBSCAN Clusters without PCA")

        return results