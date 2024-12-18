from .SupervisedModel import SupervisedModel
from sklearn.svm import SVC

class SVMModel(SupervisedModel):
    def __init__(self, data, param_grid):
        self.param_grid = param_grid
        classifiers = {
            "classifier1": SVC(random_state=42),
            "classifier2": SVC(random_state=42),
            "classifier3": SVC(random_state=42),
        }
        super().__init__(data, classifiers, param_grid)
