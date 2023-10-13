import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations as C
import math
from data import data
class LDA:
    def __init__(self, x_train, y_train, c1 = 1, c2 = 2) -> None:
        self._n_features = len(x_train[0])
        self._n_classes = len(np.unique(y_train))
        mean_overall = np.mean(np.array(x_train), axis=0)

        SP = np.zeros((self._n_features, self._n_features))
        SN = np.zeros((self._n_features, self._n_features))

        for c in np.unique(y_train):
            print(f'c = {c}')
            x_c = np.array(x_train)[y_train == c]
            print(f'x_c =\n{x_c}')
            mean_c = np.mean(x_c, axis=0)
            print(f'mean_c =\n{mean_c}')
            SP += (x_c - mean_c).T.dot((x_c - mean_c))
            print(f'SP =\n{SP}')
            mean_diff = (mean_c - mean_overall).reshape(self._n_features, 1)
            print(f'mean_diff =\n{mean_diff}')
            SN += mean_diff.dot(mean_diff.T)
            print(f'SN =\n{SN}')
        print(f'SP =\n{SP}')
        print(f'SN =\n{SN}')
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(SP).dot(SN))
        print(f'eigenvalues =\n{eigenvalues}')
        eigenvectors = eigenvectors.T
        print(f'eigenvectors =\n{eigenvectors}')
        idxs = np.argsort(abs(eigenvalues))[::-1]
        print(f'idxs =\n{idxs}')
        eigenvalues = eigenvalues[idxs]
        print(f'eigenvalues =\n{eigenvalues}')
        eigenvectors = eigenvectors[idxs]
        print(f'eigenvectors =\n{eigenvectors}')

        self._linear_discriminants = eigenvectors[:self._n_classes - 1]
        print(f'self._linear_discriminants =\n{self._linear_discriminants}')

    def transform(self, x):
        return np.dot(x, self._linear_discriminants.T)

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

D_test = LDA(np.array([[1, 1],
                       [3, 1],
                       [4, 4],
                       [8, 6]]), np.array([0, 0, 1, 1]))
D = LDA(np.array([i[2:] for i in train_datas]), np.array([len(i) for i in train_datas]))