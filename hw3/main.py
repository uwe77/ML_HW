from data import data
import re
import numpy as np
from qpsolvers import solve_qp, solve_problem, Problem


def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def score(y_true, y_pred):
    num = len(y_true)
    point = np.ones((1, num))
    return np.mean(point)


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
        print(self.w)
        print(self.b)

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        return np.sign(linear_model)


datas1 = []
datas2 = []
datas3 = []

f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(s[4])
    data_input[:] = [float(i) for i in s[:4]]
    if len(data_input) == 1:
        datas1.append(data_input)
    elif len(data_input) == 2:
        datas2.append(data_input)
    elif len(data_input) == 3:
        datas3.append(data_input)


# 将数据转换为NumPy数组
x_test = np.vstack((np.array([i[2:] for i in datas3[25:]]), np.array([i[2:] for i in datas2[25:]])))
x_train = np.vstack((np.array([i[2:] for i in datas3[:25]]), np.array([i[2:] for i in datas2[:25]])))
y_train = np.hstack((np.array([-1 for i in datas3[:25]]), np.array([1 for i in datas2[:25]])))
y_ans = np.hstack((np.array([len(i) for i in datas3[25:]]), np.array([len(i) for i in datas2[25:]])))


svm = SVM_linear()
svm.fit(x_train, y_train)
svm_ans = svm.predict(x_test)
print(svm_ans)
print(score(y_ans, svm_ans))