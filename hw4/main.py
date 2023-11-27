from data import *
from SVM import RBF_SVM as rbf
import re
import numpy as np


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
f.close()
d_space = data_space()
d_space.append(data_class(datas1, int(datas1[0])))
d_space.append(data_class(datas2, int(datas2[0])))
d_space.append(data_class(datas3, int(datas3[0])))


# 將數據轉換為numpy array
x_test = np.vstack((np.array([i[2:] for i in datas2[25:]]), np.array([i[2:] for i in datas3[25:]])))
x_train = np.vstack((np.array([i[2:] for i in datas2[:25]]), np.array([i[2:] for i in datas3[:25]])))