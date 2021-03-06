from models import linear
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import IPython

class KernelRBF(linear.Linear):


    def __init__(self, gamma):
        self.gamma = gamma

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.params(X)
        X = self.whiten(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.05)
        self.est = KernelRidge(kernel='rbf', gamma=self.gamma)
        self.est.fit(X_train, y_train)

        return self.est




