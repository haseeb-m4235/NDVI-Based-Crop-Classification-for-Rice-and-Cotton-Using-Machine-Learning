from .SupervisedModel import SupervisedModel
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier


class Bagging(SupervisedModel):
    def __init__(self, data, param_grid):
        self.grid = ParameterGrid(param_grid)
        super().__init__(data, model_name="Bagging")

    def train_single_model(self, X_train, y_train, X_test, y_test):
        results = []
        for params in tqdm(self.grid):
            # Create the base estimator with the hyperparameters from the grid
            base_estimator = DecisionTreeClassifier(max_depth=params.get("base_estimator__max_depth", None))

            # Update the params dictionary to include the base_estimator
            classifier = BaggingClassifier(
                base_estimator=base_estimator, 
                random_state=42, 
                n_jobs=-1, 
                n_estimators=params["n_estimators"]
            )
            
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.class_names)

            # Extract feature importance if the base estimator is a DecisionTree
            feature_importances = classifier.estimators_[0].feature_importances_ if hasattr(classifier.estimators_[0], 'feature_importances_') else None

            result = {
                "params": params,
                "weighted_f1": weighted_f1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion_matrix": conf_matrix,
                "classification_report": report,
                "feature_importances": feature_importances
            }

            results.append(result)

        results_df = pd.DataFrame(results)
        return results_df
