from .SupervisedModel import SupervisedModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import pandas as pd


class RandomForrest(SupervisedModel):
    def __init__(self, data, param_grid):
        self.param_grid = param_grid
        classifers = {"classifier1": RandomForestClassifier(random_state=42), "classifier2": RandomForestClassifier(random_state=42), "classifier3": RandomForestClassifier(random_state=42), }
        super().__init__(data, classifers, param_grid)
