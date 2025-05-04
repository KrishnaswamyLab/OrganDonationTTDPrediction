import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

class DummyModel(BaseEstimator, ClassifierMixin):
    """Dummy classifier returns input features as output probabilities."""
    
    def fit(self, X, y):
        # Store the number of classes from y
        self.classes_ = np.unique(y)
        return self
    
    def predict_proba(self, X):
        return X
