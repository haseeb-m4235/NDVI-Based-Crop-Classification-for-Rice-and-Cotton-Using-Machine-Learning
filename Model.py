import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

class Model():
    def __init__(self):
        self.class_names=['rice', 'cotton']
    
    def plot_confusion_matrix(self, conf_matrix, class_names):
        """
        Plot the confusion matrix using matplotlib and seaborn.
        
        Args:
            conf_matrix (array): Confusion matrix.
            class_names (list): List of class names for labeling the matrix.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def evaluate(self, y_true, y_pred,):
        """
        Evaluate the model using accuracy, F1-score, precision, recall, and confusion matrix.
        
        Args:
            y_true (list or array): Ground truth labels.
            y_pred (list or array): Predicted labels from the model.
            class_names (list): List of class names for binary classification.
        
        Returns:
            dict: A dictionary containing evaluation metrics.
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Print evaluation summary
        print("Evaluation Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(conf_matrix, self.class_names)
        
        # Return metrics as dictionary
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix
        }
        return metrics
