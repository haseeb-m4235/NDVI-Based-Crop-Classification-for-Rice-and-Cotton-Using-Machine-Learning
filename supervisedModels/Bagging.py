from .SupervisedModel import SupervisedModel
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

class Bagging(SupervisedModel):
    def __init__(self, data, param_grid):
        self.param_grid = param_grid
        classifers = {"classifier1": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,random_state=42, max_samples=0.8, max_features=0.8), "classifier2": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,random_state=42, max_samples=0.8, max_features=0.8), "classifier3": BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,random_state=42, max_samples=0.8, max_features=0.8), }
        super().__init__(data, classifers, param_grid)