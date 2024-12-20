import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from scipy.optimize import linear_sum_assignment

class UnsupervisedModel:
    def __init__(self, data):
        """
        Initialize the UnsupervisedModel class.

        Args:
            data (dict): Dictionary containing datasets (e.g., X_train, y_train, etc.).
        """
        self.data = data

    def apply_pca(self, X, n_components=2):
        """
        Apply PCA for dimensionality reduction.

        Args:
            X (array-like): Dataset to reduce.
            n_components (int): Number of principal components.

        Returns:
            tuple: Reduced dataset and PCA object.
        """
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        return X_reduced, pca

    def evaluate_clustering(self, y_true, y_pred):
        """
        Evaluate clustering performance using cluster purity and confusion matrix.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted cluster labels.

        Returns:
            dict: Dictionary containing purity, confusion matrix, and classification report.
        """
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Optimize cluster assignments using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)
        optimal_matrix = conf_matrix[row_ind[:, None], col_ind]

        # Calculate cluster purity
        purity = np.sum(optimal_matrix.diagonal()) / np.sum(conf_matrix)

        # Print results
        print(f"Cluster Purity: {purity:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred)) #target_names = ['Rice', 'Cotton']

        # Plot confusion matrix
        class_names = ['Rice', 'Cotton']
        self.plot_confusion_matrix(conf_matrix,class_names)

        return {
            "purity": purity,
            "confusion_matrix": conf_matrix,
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
        }

    def plot_confusion_matrix(self, conf_matrix,class_names):
        """
        Plot the confusion matrix using matplotlib and seaborn.

        Args:
            conf_matrix (array-like): Confusion matrix to plot.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_clusters(self, X, y_pred, title="Clusters"):
        """
        Plot the clusters.

        Args:
            X (array-like): Dataset (after dimensionality reduction if applicable).
            y_pred (array-like): Predicted cluster labels.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred, palette="tab10", legend='full')
        plt.title(title)
        plt.show()