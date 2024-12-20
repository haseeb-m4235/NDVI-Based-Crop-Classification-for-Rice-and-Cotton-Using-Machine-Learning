from .SupervisedModel import SupervisedModel
from sklearn.ensemble import RandomForestClassifier
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

class RandomForrest(SupervisedModel):
    def __init__(self, data, param_grid):
        self.grid = ParameterGrid(param_grid)
        super().__init__(data, model_name="RandomForrest")

    def train_single_model(self, X_train, y_train, X_test, y_test):
        results = []
        for params in tqdm(self.grid):
            classifier = RandomForestClassifier(random_state=42,n_jobs=-1, **params)
            classifier.fit(X_train, y_train)
            importances = classifier.feature_importances_

            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.class_names)


            result = {"params":params, "weighted_f1":weighted_f1, "accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "confusion_matrix":conf_matrix, "classification_report":report, "feature_importances":importances}
            # for key, value in result.items():
            #     print(f"{key}: {value}\n")
            
            results.append(result)
            # print(result)

        results_df = pd.DataFrame(results)
        return results_df
