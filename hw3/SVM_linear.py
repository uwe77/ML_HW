import numpy as np
from qpsolvers import solve_qp, solve_problem, Problem


def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)



class SVM_linear:
    def __init__(self, C=1.0):
        self.C = C
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = linear_kernel(X, X)
        P = np.outer(y, y) * K
        q = -np.ones((n_samples, 1))
        A = y.astype(np.double)
        b = np.array([0.0])
        lb = np.zeros(n_samples).astype(np.double)
        ub = np.ones(n_samples)* self.C
        problem = Problem(P, q, None, None, A, b, lb, ub)
        solution = solve_problem(problem, solver='proxqp')
        alpha = np.array([round(i, 4) for i in solution.x])
        self.w = np.array([round(i, 4) for i in np.dot(X.T, alpha * y)])
        tmp_x = np.array(X)[np.logical_and(0 < alpha, alpha < self.C), :]
        tmp_y = np.array(y)[np.logical_and(0 < alpha, alpha < self.C)]
        self.b = round(np.mean(tmp_y - np.dot(tmp_x, self.w)), 4)
        print("alpha:\n", alpha, "\nand sum: ", round(sum(alpha), 4))
        print("weight: ",self.w)
        print("bias: ",self.b)

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return np.sign(linear_model)