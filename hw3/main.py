from data import data
import re
import numpy as np
from qpsolvers import solve_qp

class SVM_linear:
    def __init__(self, C=1.0): # C is the penalty parameter of the error term
        self.C = C # C is the penalty parameter of the error term
        self.w = None # weights
        self.b = None # bias

    def fit(self, X, y): # X is the training data, y is the target values
        num_samples, num_features = X.shape # num_samples is the number of samples, num_features is the number of features

        # Set up the quadratic programming problem
        P = np.zeros((num_features + 1, num_features + 1)) # P is the matrix of the quadratic objective function
        P[:num_features, :num_features] = np.eye(num_features) # P is the identity matrix
        print("P=\n",P)
        q = np.zeros(num_features + 1) # q is the vector of the quadratic objective function
        q[-1] = self.C # q is the penalty parameter of the error term
        print("q=\n",q)
        
        G = -np.diag(y) @ np.hstack((X, np.ones((num_samples, 1)))) # G is the matrix of the inequality constraints
        print("G=\n",G)
        h = -np.ones(num_samples) # h is the vector of the inequality constraints
        print("h=\n",h)
        # Solve the quadratic programming problem
        alpha = solve_qp(P = P, q = q, G = G, h = h) # alpha is the Lagrange multipliers
        print("alpha=\n",alpha)
        # Calculate the weights and bias
        self.w = X.T @ (alpha[:-1] * y) # w is the weights
        print(self.w)
        self.b = np.mean(y - X @ self.w) # b is the bias
        print(self.b)

    def predict(self, X): # X is the test data
        linear_model = X @ self.w + self.b # linear_model is the linear model
        return np.sign(linear_model) # return the sign of the linear model
    

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

# datas1 = np.array(datas1)
# datas2 = np.array(datas2)
# datas3 = np.array(datas3)

# 将数据转换为NumPy数组
x_test = np.vstack((np.array([i[2:] for i in datas3[25:]]), np.array([i[2:] for i in datas2[25:]])))
x_train = np.vstack((np.array([i[2:] for i in datas3[:25]]), np.array([i[2:] for i in datas2[:25]])))
y_train = np.hstack((np.array([-1 for i in datas3[:25]]), np.array([1 for i in datas2[:25]])))
y_ans = np.hstack((np.array([len(i) for i in datas3[25:]]), np.array([len(i) for i in datas2[25:]])))
X = x_train
y = y_train

# 将数据转换为特征矩阵和标签向量

svm = SVM_linear()
svm.fit(X, y)

# 训练SVM模型

print(svm.w, svm.b)

# 输出权重和偏置

