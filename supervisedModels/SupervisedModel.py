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
import pandas as pd

class SupervisedModel():
    def __init__(self, data):
        self.class_names=['rice', 'cotton']
        self.data = data

    def train_single_model(self, X_train, y_train, X_test, y_test):
        pass
    
    def get_best_hyperparameters(self):
        results = []
        for i in range(1, 4):
            X_train = self.data.get(f"X_train_{i}")
            y_train = self.data.get(f"y_train_{i}")
            X_test = self.data.get(f"X_test_{i}")
            y_test = self.data.get(f"y_test_{i}")
            print(f"\n\nResults on testing set 202{i}")
            result = self.train_single_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
            results.append(result)

        combined_df = pd.concat(results, ignore_index=True)
        combined_df["params_str"] = combined_df["params"].apply(lambda x: str(x))

        averaged_scores = combined_df.groupby("params_str", as_index=False)["weighted_f1"].mean()
        highest_params = averaged_scores.loc[averaged_scores["weighted_f1"].idxmax()]
        
        self.best_params = eval(highest_params["params_str"])
        self.individual_scores = combined_df[combined_df["params_str"] == highest_params["params_str"]]
        
        # Display the results
        print("Parameters with Highest Mean Weighted F1 Score:")
        print(f"Params: {self.best_params}")
        print(f"Highest mean weighted F1 Score: {highest_params['weighted_f1']}\n")
        return self.best_params
    
    def get_test_results(self):
        print(f"\n\nindividual scroe: {self.individual_scores}\n\n")
        years = [2021, 2022, 2023]
        count=0
        for index, row in self.individual_scores.iterrows():
            print(f"Final Results for year: {years[count]}:")
            print(f"Parameters: {row['params']}")
            print(f"Weighted F1: {row['weighted_f1']:.4f}")
            print(f"Accuracy: {row['accuracy']:.4f}")
            print(f"Precision: {row['precision']:.4f}")
            print(f"Recall: {row['recall']:.4f}")
            print(f"F1 Score: {row['f1']:.4f}")
            print(f"Confusion Matrix: {row['confusion_matrix']}")
            print(f"Classification Report: {row['classification_report']}\n")
            count+=1
        
        # Plot confusion matrix for each year
        count=0
        for index, row in self.individual_scores.iterrows():
            cm = np.array(row['confusion_matrix'])
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=["Predicted 0", "Predicted 1"], 
                        yticklabels=["Actual 0", "Actual 1"])
            plt.title(f"Confusion Matrix for {years[count]}")
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.show()
            count+=1

        # for i, year in enumerate(years):
        #     print(f"Final Results for year: {year}:")
        #     print(f"Parameters: {self.individual_scores.iloc[i, 'params']}")
        #     print(f"Weighted F1: {self.individual_scores.iloc[i, 'weighted_f1']:.4f}")
        #     print(f"Accuracy: {self.individual_scores.iloc[i, 'accuracy']:.4f}")
        #     print(f"Precision: {self.individual_scores.iloc[i, 'precision']:.4f}")
        #     print(f"Recall: {self.individual_scores.iloc[i, 'recall']:.4f}")
        #     print(f"F1 Score: {self.individual_scores.iloc[i, 'f1']:.4f}")
        #     print(f"Confusion Matrix: {self.individual_scores.iloc[i, 'confusion_matrix']}")
        #     print(f"Classification Report: {self.individual_scores.iloc[i, 'classification_report']}\n")
        
        # # Plot confusion matrix for each year
        # for i, year in enumerate(years):
        #     cm = np.array(self.individual_scores.iloc[i, 'confusion_matrix'])
            
        #     plt.figure(figsize=(6, 5))
        #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        #                 xticklabels=["Predicted 0", "Predicted 1"], 
        #                 yticklabels=["Actual 0", "Actual 1"])
        #     plt.title(f"Confusion Matrix for {year}")
        #     plt.xlabel('Predicted Labels')
        #     plt.ylabel('True Labels')
        #     plt.show()
    