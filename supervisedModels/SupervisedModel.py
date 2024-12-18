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
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import f1_score, make_scorer

class SupervisedModel():
    def __init__(self, data, classifiers, param_grid):
        self.class_names=['rice', 'cotton']
        self.data = data
        self.param_grid = param_grid
        self.classifiers = classifiers
        self.scorer = make_scorer(f1_score, average='weighted')

    def train_single_model(self, X_train, y_train, classifier):
        grid_search = GridSearchCV(estimator=classifier, param_grid=self.param_grid, cv=3, verbose=2,n_jobs=-1, scoring=self.scorer)
        grid_search.fit(X_train, y_train)
        results = pd.DataFrame(grid_search.cv_results_)
        results = results[['params', 'mean_test_score']]
        return results
    
    def get_best_hyperparameters(self):
        results = []
        for i in range(1, 4):
            X_train = self.data.get(f"X_train_{i}")
            y_train = self.data.get(f"y_train_{i}")
            classifer = self.classifiers.get(f"classifier{i}")
            results.append(self.train_single_model(X_train, y_train, classifer))

        # Combine all results into a single DataFrame
        combined_df = pd.concat(results, ignore_index=True)
        
        # Convert `params` dictionaries to a hashable string representation
        combined_df["params_str"] = combined_df["params"].apply(lambda x: str(x))
        
        # Group by the string representation and calculate the mean test score
        averaged_scores = combined_df.groupby("params_str", as_index=False)["mean_test_score"].mean()
        
        # Find the row with the highest mean test score
        highest_params = averaged_scores.loc[averaged_scores["mean_test_score"].idxmax()]
        
        # Retrieve the original `params` dictionary
        best_params = eval(highest_params["params_str"])  # Convert string back to dictionary
        
        # Get individual scores for the best parameters
        individual_scores = combined_df[combined_df["params_str"] == highest_params["params_str"]]
        
        # Display the results
        print("Parameters with Highest Mean Test Score:")
        print(f"Params: {best_params}")
        print(f"Highest Mean Test Score: {highest_params['mean_test_score']}")
        print("\nIndividual Mean Test Scores:")
        print(individual_scores[["mean_test_score"]])
        return best_params

    
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
