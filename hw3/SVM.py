import numpy as np
from qpsolvers import solve_qp, solve_problem, Problem

#=========================================Linear_SVM==========================================================
class Linear_SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.alpha = None
        self.b = None
        self.x_train = None
        self.y_train = None

    def kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.x_train = X
        self.y_train = y
        K = self.kernel(X, X)
        P = np.outer(y, y) * K
        q = -np.ones((n_samples, 1))
        A = y.astype(np.double)
        b = np.array([0.0])
        lb = np.zeros(n_samples).astype(np.double)
        ub = np.ones(n_samples)* self.C
        problem = Problem(P, q, None, None, A, b, lb, ub)
        solution = solve_problem(problem, solver='proxqp')
        alpha = np.array([round(i, 4) for i in solution.x])
        self.alpha = alpha
        self.b = round(np.mean(y - np.dot(K, alpha * y)), 4)
        # tmp_x = np.array(X)[np.logical_and(0 < alpha, alpha < self.C), :]
        # tmp_y = np.array(y)[np.logical_and(0 < alpha, alpha < self.C)]
        # tmp_alpha = alpha[np.logical_and(0 < alpha, alpha < self.C)]
        # # self.b = round(np.mean(tmp_y - np.dot(tmp_x, self.w)), 4)
        # self.b = round(np.mean(tmp_y - np.dot(self.kernel(tmp_x, tmp_x), tmp_alpha * tmp_y)), 4)
        # print("alpha:\n", alpha, "\nand sum: ", round(sum(alpha), 4))
        print("> bias: ",self.b)

    def predict(self, X):
        linear_model = np.array([])
        for i in X:
            linear_model = np.append(linear_model, np.dot(self.kernel(self.x_train, i), self.alpha * self.y_train) + self.b)
        return np.sign(linear_model)
#=====================================RBF_SVM==============================================================    
class RBF_SVM(Linear_SVM):
    def __init__(self, C=1.0, sigama=1.0):
        self.sigama = sigama
        super().__init__(C)

    def kernel(self, x1, x2):
        n_samples1, n_features1 = x1.shape
        try:
            n_samples2, n_features2 = x2.shape
        except:
            x2 = np.array([x2])
            n_samples2, n_features2 = x2.shape
        K = np.zeros((n_samples1, n_samples2))
        for i in range(n_samples1):
            for j in range(n_samples2):
                K[i, j] = np.exp(-np.linalg.norm(x1[i] - x2[j])**2 / (2 * self.sigama**2))
        K = K.T
        return K
#=====================================poly_SVM==============================================================
class poly_SVM(Linear_SVM):
    def __init__(self, C, p) -> None:
        self.p = p
        super().__init__(C)
    
    def kernel(self, x1, x2):
        K = (np.dot(x1, x2.T)) ** self.p
        return K