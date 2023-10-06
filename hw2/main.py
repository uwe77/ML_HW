import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations as C
class data:
    def __init__(self, y=0) -> None:
        self._feature = []
        self._label = int(y)

    def __setitem__(self, index, value)->None:
        self._feature[index] = value

    def __getitem__(self, index)->None:
        return self._feature[index]

    def __len__(self):
        return self._label

    def append(self, x):
        self._feature.append(float(x))

    def __str__(self) -> str:
        return f'{self._feature},{self._label}'

    def set_label(self, y = 0):
        self._label = int(y)
        return self

class LDA:
    def __init__(self, x_train, y_train) -> None:
        self._num_posi = 0
        self._num_nage = 0
        self._m_posi = np.zeros(1, np.array(x_train).shape[1])
        self._m_nage = np.zeros(1, np.array(x_train).shape[1])

        for i in range(len(x_train)):
            if y_train[i]:
                self._num_posi += 1
                self._m_posi += np.array(x_train[i])
            else:
                self._num_nage += 1
                self._m_nage += np.array(x_train[i])
        self._m_posi /= self._num_posi
        self._m_nage /= self._num_nage
        self._p_posi = self._num_posi / (self._num_nage+ self._num_posi)
        self._p_nage = self._num_nage / (self._num_nage+ self._num_posi)
        

datas_label1 = []
datas_label2 = []
datas_label3 = []

f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(s[4])
    data_input[:] = [float(i) for i in s[:4]]
    if len(data_input) == 1:
        datas_label1.append(data_input)
    elif len(data_input) == 2:
        datas_label2.append(data_input)
    elif len(data_input) == 3:
        datas_label3.append(data_input)

train_datas = []
test_datas = []
for i in range(25):
    train_datas.append(datas_label2[i].set_label(1))
    train_datas.append(datas_label3[i].set_label(0))
for i in range(25,50):
    test_datas.append(datas_label2[i].set_label(1))
    test_datas.append(datas_label3[i].set_label(0))

m1, m2 = np.zeros(2)
for i in train_datas:
    if len(i):
        m1 += np.array(i[2:])
    else:
        m2 += np.array(i[2:])
m1 /= (len(train_datas))/2
m2 /= (len(train_datas))/2

