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
        self._feature[index] = float(value)

    def __getitem__(self, index)->None:
        return self._feature[index]

    def __len__(self):
        return self._label

    def append(self, x):
        self._feature.append(float(x))

    def __str__(self) -> str:
        return f'{self._feature},{self._label}'

    def label(self, y = 0):
        self._label = int(y)

datas_label1 = []
datas_label2 = []
datas_label3 = []

f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(s[4])
    for i in range(4):
        data_input.append(float(s[i]))
    if len(data_input) == 1:
        datas_label1.append(data_input)
    elif len(data_input) == 2:
        datas_label2.append(data_input)
    elif len(data_input) == 3:
        datas_label3.append(data_input)

train_datas = []
test_datas = []
for i in range(25):
    train_datas.append(datas_label1[i])
    train_datas.append(datas_label2[i])
for i in range(25,50):
    test_datas.append(datas_label1[i])
    test_datas.append(datas_label2[i])
