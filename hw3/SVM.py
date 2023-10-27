import numpy as np
from qpsolvers import solve_qp, solve_problem, Problem

#=========================================Linear_SVM==========================================================
def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

class Linear_SVM:
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
#=====================================RBF_SVM==============================================================    

def RBF_kernel(x1, x2, sigama=1.0):
    n_samples1, n_features1 = x1.shape
    n_samples2, n_features2 = x2.shape
    print(n_samples2)
    K = np.zeros((n_samples1, n_samples2))
    for i in range(n_samples1):
        for j in range(n_samples2):
            K[i, j] = np.exp(-np.linalg.norm(x1[i] - x2[j])**2 / (2 * sigama**2))
    return K
class RBF_SVM(Linear_SVM):
    def __init__(self, C=1.0, sigama=1.0):
        self.sigama = sigama
        super().__init__(C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = RBF_kernel(X, X, self.sigama)
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