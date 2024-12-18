from ..Model import Model
from sklearn.ensemble import RandomForestClassifier

class RandomForrest(Model):
    def __init__(self, X_train, y_train, X_test, y_test):
        super(X_train, y_train, X_test, y_test)
        self.classifier = RandomForestClassifier()
