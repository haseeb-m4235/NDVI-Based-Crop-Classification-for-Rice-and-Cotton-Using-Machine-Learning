from .SupervisedModel import SupervisedModel
from xgboost import XGBClassifier

class XGBoostModel(SupervisedModel):
    def __init__(self, data, param_grid):
        self.param_grid = param_grid
        classifiers = {
            "classifier1": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "classifier2": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "classifier3": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        }
        super().__init__(data, classifiers, param_grid)
