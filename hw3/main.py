from data import data
import re
import numpy as np
from qpsolvers import solve_qp, solve_problem, Problem
from SVM_linear import SVM_linear

    
def score(y_true, y_pred):
    score_board = np.array([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
    return np.mean(score_board)*100
    # pass

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
y_ans = np.hstack((np.array([-1 for i in datas3[25:]]), np.array([1 for i in datas2[25:]])))


c_rate = np.array([])
for i in [1, 10, 100]:
    svm = SVM_linear(C=i)
    svm.fit(x_train, y_train)
    svm_ans = svm.predict(x_test)
    c_rate = np.append(c_rate, score(y_ans, svm_ans))
    print("score_C", i, ": ", score(y_ans, svm_ans))
    svm.fit(x_test, y_ans)
    svm_ans = svm.predict(x_train)
    c_rate = np.append(c_rate, score(y_ans, svm_ans))
    print("score_C", i, ": ", score(y_ans, svm_ans))
c_rate = np.reshape(c_rate, (3, 2))
c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(3)])
print("c_rate: ", c_rate)