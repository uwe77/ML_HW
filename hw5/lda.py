import numpy as np
import math

class LDA:
    def __init__(self, C1=1, C2=1):
        self._c1 = C1
        self._c2 = C2

    def fit(self, X, y):
        S = np.zeros((X.shape[1], X.shape[1]))
        self._n_f = X.shape[1]
        self._n_classes = len(np.unique(y))
        self._class_type = []
        SP = np.zeros((self._n_f, self._n_f))
        SN = np.zeros((self._n_f, self._n_f))
        m1 = None
        m2 = None
        for c in np.unique(y):
            self._class_type.append(c)
            x_c = np.array(X)[y == c]
            mean_c = np.mean(x_c, axis=0)
            for i in x_c:
                diff = i-mean_c
                m = np.zeros((2, len(diff)))
                m[0] = diff
                m = m.T.dot(m)
                if c:
                    SP += m
                else:
                    SN += m
            if c:
                SP /= x_c.shape[0]-1
                S += (SP*x_c.shape[0])
                m1 = mean_c
            else:
                SN /= x_c.shape[0]-1
                S += SN*x_c.shape[0]
                m2 = mean_c
        S /= X.shape[0]
        self._w = np.array([round(i,2) for i in(np.dot((m1-m2), np.linalg.inv(S)))])
        self._b = round(-0.5*self._w.dot(m1+m2) - math.log(self._c1/self._c2), 2)
    def predict(self, X):
        y = []
        for i in X:
            pos = np.dot(i, self._w.T)+self._b
            if pos > 0: # y=1
                y.append(1)
            elif pos <= 0: # y = 0
                y.append(0)
        return np.array(y)

