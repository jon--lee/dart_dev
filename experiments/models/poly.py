from models import linear
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import IPython

class Poly(linear.Linear):


    def __init__(self, degree):
        self.degree = degree
        self.transform = PolynomialFeatures(degree=degree)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.params(X)
        X = self.whiten(X)
        X_ = self.transform.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=.05)
        self.est = LinearRegression()
        self.est.fit(X_train, y_train)

        return self.est

    def predict(self, X):
        X = np.array(X)
        X = self.whiten(X)
        X_ = self.transform.fit_transform(X)
        return self.est.predict(X_)

    def score(self, X, y):
        preds = self.predict(X)
        return r2_score(y, preds)




