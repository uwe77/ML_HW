from data import data
import re
import numpy as np
from SVM import Linear_SVM, RBF_SVM, poly_SVM
from qpsolvers import solve_qp, solve_problem, Problem


def score(y_true, y_pred):
    score_board = np.array([1 if y_true[i] == y_pred[i] else 0 for i in range(len(y_true))])
    return np.mean(score_board)*100

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

# 將數據轉換為numpy array
x_test = np.vstack((np.array([i[2:] for i in datas2[25:]]), np.array([i[2:] for i in datas3[25:]])))
x_train = np.vstack((np.array([i[2:] for i in datas2[:25]]), np.array([i[2:] for i in datas3[:25]])))
y_train = np.hstack((np.array([1 for i in datas2[:25]]), np.array([-1 for i in datas3[:25]])))
y_ans = np.hstack((np.array([1 for i in datas2[25:]]), np.array([-1 for i in datas3[25:]])))

# =========================================Linear_SVM==========================================================
c_rate = np.array([])
for i in [1, 10, 100]:
    svm = Linear_SVM(C=i)
    svm.fit(x_train, y_train)
    svm_ans = svm.predict(x_test)
    c_rate = np.append(c_rate, score(y_ans, svm_ans))
    print("score_C", i, "_1: ", score(y_ans, svm_ans))
    svm.fit(x_test, y_ans)
    svm_ans = svm.predict(x_train)
    c_rate = np.append(c_rate, score(y_train, svm_ans))
    print("score_C", i, "_2: ", score(y_train, svm_ans))
c_rate = np.reshape(c_rate, (3, 2))
c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(3)])
print("c_rate: ", c_rate)
#=========================================RBF_SVM=============================================================
c_rate = np.array([])
svm = RBF_SVM(C=10, sigama=5)
svm.fit(x_train, y_train)
svm_ans = svm.predict(x_test)
c_rate = np.append(c_rate, score(y_ans, svm_ans))
print("score_C10_p", 1, "_1: ", score(y_ans, svm_ans))
svm.fit(x_test, y_ans)
svm_ans = svm.predict(x_train)
c_rate = np.append(c_rate, score(y_train, svm_ans))
print("score_C10_p", 1, "_2: ", score(y_train, svm_ans))
c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(2)])
print("c_rate: ", c_rate)

c_rate = np.array([])
for i in [1, 10, 100]:
    for j in [1, 0.5, 0.1, 0.05]:
        print("- C",i,", sigma:",j)
        print("> ---1---")
        svm = poly_SVM(C=i, p=j)
        svm.fit(x_train, y_train)
        svm_ans = svm.predict(x_test)
        c_rate = np.append(c_rate, score(y_ans, svm_ans))
        print("> score: ", score(y_ans, svm_ans),"\n")
        print("> ---2---")
        svm.fit(x_test, y_ans)
        svm_ans = svm.predict(x_train)
        c_rate = np.append(c_rate, score(y_train, svm_ans))
        print("score: ",score(y_train, svm_ans))
c_rate = np.reshape(c_rate, (12, 2))
c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(12)])
print("\n> c_rate: ", c_rate)
#=========================================poly_SVM============================================================
c_rate = np.array([])
svm = poly_SVM(C=10, p=1)
svm.fit(x_train, y_train)
svm_ans = svm.predict(x_test)
c_rate = np.append(c_rate, score(y_ans, svm_ans))
print("score_C10_p", 1, "_1: ", score(y_ans, svm_ans))
svm.fit(x_test, y_ans)
svm_ans = svm.predict(x_train)
c_rate = np.append(c_rate, score(y_train, svm_ans))
print("score_C10_p", 1, "_2: ", score(y_train, svm_ans))
print("c_rate: ", c_rate)
c_rate = np.array([])
for i in [1, 10, 100]:
    print("\n")
    for j in [2, 3, 4, 5]:
        print("- C",i,", p:",j)
        print("> ---1---")
        svm = poly_SVM(C=i, p=j)
        svm.fit(x_train, y_train)
        svm_ans = svm.predict(x_test)
        c_rate = np.append(c_rate, score(y_ans, svm_ans))
        print("> score: ", score(y_ans, svm_ans),"\n")
        print("> ---2---")
        svm.fit(x_test, y_ans)
        svm_ans = svm.predict(x_train)
        c_rate = np.append(c_rate, score(y_train, svm_ans))
        print("score: ",score(y_train, svm_ans))
c_rate = np.reshape(c_rate, (12, 2))
c_rate = np.array([round(np.mean(c_rate[i]), 2) for i in range(12)])
print("\n> c_rate: ", c_rate)
