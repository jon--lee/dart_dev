from models import estimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

class Linear(estimator.Estimator):


    def __init__(self):
        return

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.params(X)
        X = self.whiten(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05)
        self.est = LinearRegression()
        self.est.fit(X_train, y_train)

        return self.est

    def predict(self, X):
        X = np.array(X)
        X = self.whiten(X)
        return self.est.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)




