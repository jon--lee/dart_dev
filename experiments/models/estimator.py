import numpy as np

class Estimator():

    def whiten(self, X):
        if self.mean is None or self.std is None:
            raise Exception("Called whiten but mean or std not set yet.")

        X = X - self.mean
        X = X / self.std
        locs = np.isnan(X)
        X[locs] = 0.0
        locs = np.isinf(X)
        X[locs] = 0.0
        return X

    def params(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError

    def score(self, X, y):
        raise NotImplementedError