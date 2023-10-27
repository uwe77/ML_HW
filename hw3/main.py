from data import data
import re
import numpy as np
from SVM import Linear_SVM, RBF_SVM
from qpsolvers import solve_qp, solve_problem, Problem


def score(y_true, y_pred):
    score_board = np.array([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
    return np.mean(score_board)*100
    # pass

def poly_kernel(x1, x2, p=1):
    print(((np.dot(x1, x2.T) + 1) ** p).shape)
    return (np.dot(x1, x2.T) + 1) ** p
class poly_SVM(Linear_SVM):
    def __init__(self, C, p) -> None:
        self.p = p
        super().__init__(C)
    
    def fit(self, X, y):
        self.y_train = y
        n_samples, n_features = X.shape
        K = poly_kernel(X, X, self.p)
        n_samples = K.shape[0]
        P = np.outer(y, y) * K
        q = -np.ones((n_samples, 1))
        A = y.astype(np.double)
        b = np.array([0.0])
        lb = np.zeros(n_samples).astype(np.double)
        ub = np.ones(n_samples)* self.C
        problem = Problem(P, q, None, None, A, b, lb, ub)
        solution = solve_problem(problem, solver='proxqp')
        self.alpha = np.array([round(i, 4) for i in solution.x])
        self.w = np.array([round(i, 4) for i in np.dot(X.T, self.alpha * y)])
        tmp_x = np.array(X)[np.logical_and(0 < self.alpha, self.alpha < self.C), :]
        tmp_y = np.array(y)[np.logical_and(0 < self.alpha, self.alpha < self.C)]
        self.b = round(np.mean(tmp_y - np.dot(tmp_x, self.w)), 4)
        print("alpha:\n", self.alpha, "\nand len: ", len(self.alpha), "sum: ", round(sum(self.alpha), 4))
        print("weight: ",self.w)
        print("bias: ",self.b)

    def predict(self, X):
        K = poly_kernel(X, X, self.p)
        return np.sign(np.dot(K, self.alpha * self.y_train) + self.b)


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
x_test = np.vstack((np.array([i[2:] for i in datas2[25:]]), np.array([i[2:] for i in datas3[25:]])))
x_train = np.vstack((np.array([i[2:] for i in datas2[:25]]), np.array([i[2:] for i in datas3[:25]])))
y_train = np.hstack((np.array([1 for i in datas2[:25]]), np.array([-1 for i in datas3[:25]])))
y_ans = np.hstack((np.array([1 for i in datas2[25:]]), np.array([-1 for i in datas3[25:]])))

svm = poly_SVM(C=10, p=2)
svm.fit(x_train, y_train)
svm_ans = svm.predict(x_test)
print("score: ", score(y_ans, svm_ans))

# #=========================================Linear_SVM==========================================================
# c_rate = np.array([])
# for i in [1, 10, 100]:
#     svm = Linear_SVM(C=i)
#     svm.fit(x_train, y_train)
#     svm_ans = svm.predict(x_test)
#     c_rate = np.append(c_rate, score(y_ans, svm_ans))
#     print("score_C", i, "_1: ", score(y_ans, svm_ans))
#     svm.fit(x_test, y_ans)
#     svm_ans = svm.predict(x_train)
#     c_rate = np.append(c_rate, score(y_ans, svm_ans))
#     print("score_C", i, "_2: ", score(y_ans, svm_ans))
# c_rate = np.reshape(c_rate, (3, 2))
# c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(3)])
# print("c_rate: ", c_rate)
# #=========================================RBF_SVM=============================================================
# c_rate = np.array([])
# for i in [1, 10, 100]:
#     for j in [1, 0.5, 0.1, 0.05]:
#         svm = RBF_SVM(C=i, sigama=j)
#         svm.fit(x_train, y_train)
#         svm_ans = svm.predict(x_test)
#         c_rate = np.append(c_rate, score(y_ans, svm_ans))
#         print("score_C", i, "_sigama", j, ": ", score(y_ans, svm_ans))
#         svm.fit(x_test, y_ans)
#         svm_ans = svm.predict(x_train)
#         c_rate = np.append(c_rate, score(y_ans, svm_ans))
#         print("score_C", i, "_sigama", j, ": ", score(y_ans, svm_ans))
# c_rate = np.reshape(c_rate, (12, 2))
# c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(12)])
# print("c_rate: ", c_rate)