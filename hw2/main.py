import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations as C
import math
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
    def __init__(self, x_train, y_train, cin1=1, cin2=1) -> None:
        num_posi = 0
        num_nage = 0
        dim1x2 = (1, int(np.array(x_train).shape[1]))
        m_posi = np.zeros(dim1x2)
        m_nage = np.zeros(dim1x2)
        self._c1 = cin1
        self._c2 = cin2

        for i in range(len(x_train)):
            if y_train[i]:
                num_posi += 1
                m_posi += np.array(x_train[i])
            else:
                num_nage += 1
                m_nage += np.array(x_train[i])
        m_posi /= num_posi
        m_nage /= num_nage
        p_posi = num_posi / (num_nage+ num_posi)
        p_nage = num_nage / (num_nage+ num_posi)
        dim2x2 = (np.array(x_train).shape[1], np.array(x_train).shape[1])
        s_posi = np.zeros(dim2x2)
        s_nage = np.zeros(dim2x2)
        for i in range(num_nage+num_posi):
            if y_train[i]:
                temp_2x1 = np.array(x_train[i]) - m_posi
                s_posi += np.matmul(temp_2x1, np.transpose(temp_2x1))
            else:
                temp_2x1 = np.array(x_train[i]) - m_nage
                s_nage += np.matmul(temp_2x1, np.transpose(temp_2x1))
        self._segama = s_nage*p_nage+ s_posi*p_posi

        self._w_t = np.matmul(np.transpose(m_posi-m_nage),np.linalg.inv(self._segama))
        self._b = np.matmul(self._w_t, (m_posi+m_nage))/(-2) - math.log(self.c1*p_nage/(self._c2*p_posi))

        print(f'self._w_t = \n{self._w_t}\nself._b = \n{self._b}')
        


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

D = LDA([i[2:] for i in train_datas], [len(i) for i in train_datas], 1, 1)