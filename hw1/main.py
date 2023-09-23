import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class KNN:
    def __str__(self) -> str:
        return f'k = {self._k}, feature_num = {self._features}'

    def __init__(self, F, K) -> None:
        self._k = K
        self._features = F

    def _distant(self, n1, n2):
        dis = 0
        if self._features == 1:
            dis += ((n1 - n2)**2)
        else:
            for i in range(self._features - 1):
                dis += (n1[i] - n2[i])**2
        return dis**0.5
    
    def predict(self, x_test, x_train, y):
        dis = 0
        distances = [self._distant(x_test, i) for i in x_train]
        k_indices = np.argsort(distances)[:self._k]
        k_nearest_labels = [y[i] for i in k_indices]
        prediction = np.bincount(k_nearest_labels).argmax()
        return prediction


y = []
x_input = []
f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    temp_list = []
    for i in range(4):
        temp_list.append(float(s[i]))
    y.append(int(s[4]))
    x_input.append(temp_list)

x_input = np.array(x_input)
for i in range(4):
    for j in range(i+1, 4):
        plt.scatter(x_input[:,i], x_input[:,j])
        plt.savefig(f'feature{i}_{j}.png')
        plt.clf()

K1_1 = KNN(1,1)
K2_1 = KNN(2,1)
K3_1 = KNN(3,1)
K4_1 = KNN(4,1)
x_test = x_input[75:]
x_train = x_input[:75]
y_label = y[:75]
cunt = 0
for i in range(4):
    for j in range(i,4):
        for s in range(1,4):
            if s > abs(i-j):
                for x in x_test[0, i:j:s]:
                    cunt += 1
                    k = KNN((abs(i-j) // s)+1,1)
                    print(f'knn:{k}, [i:j:s]=[{i}:{j}:{s}]')
print(f'count = {cunt}')
