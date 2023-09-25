import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations as C
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
            for i in range(self._features):
                dis += (n1[i] - n2[i])**2
        return dis**0.5
    
    def predict(self, x_test, x_train, y):
        distances = [self._distant(x_test, i) for i in x_train]
        k_indices = np.argsort(distances)[:self._k].ravel()
        k_nearest_labels = [y[i] for i in k_indices]
        prediction = np.bincount(k_nearest_labels)
        prediction = prediction.argmax()
        return prediction
def score_list(x_test, x_train, y_label, y_ans, k_num):
    scores = []
    for feature in range(1,5):
        col = list(C(range(4),feature))
        knn = KNN(feature,k_num)
        for i in range(len(col)):
            
            x_train_data = x_train[:,0]
            x_test_data = x_test[:,0]
            score = 0
            # print(f'=============={col[i]}================')
            for j in range(feature):
                x_train_data = np.c_[x_train_data, x_train[:,col[i][j]]]
                x_test_data = np.c_[x_test_data, x_test[:,col[i][j]]]

            x_train_data = x_train_data[:,1:]
            x_test_data = x_test_data[:,1:]

            for i in range(x_test_data.shape[0]):
                if knn.predict(x_test_data[i], x_train_data, y_label) == y_ans[i]:
                    score += 1
            # print(f'score:{round(score*100/75,2)}')
            scores.append(round(score*100/75,2))
    return scores

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
y = np.array(y)
x_input = np.array(x_input)
for i in range(4):
    for j in range(i+1, 4):
        plt.scatter(x_input[:,i], x_input[:,j])
        plt.savefig(f'feature{i}_{j}.png')
        plt.clf()

# x_test = x_input[75:]
# y_ans = y[75:]
# x_train = x_input[:75]
# y_label = y[:75]
print(score_list(x_input[75:], x_input[:75], y[:75], y[75:], 1))
print(score_list(x_input[:75], x_input[75:], y[75:], y[:75], 1))
print(score_list(x_input[75:], x_input[:75], y[:75], y[75:], 3))
print(score_list(x_input[:75], x_input[75:], y[75:], y[:75], 3))
# for feature in range(1,5):
#     col = list(C(range(4),feature))
#     knn = KNN(feature,1)
#     for i in range(len(col)):
        
#         x_train_data = x_train[:,0]
#         x_test_data = x_test[:,0]
#         score = 0
#         print(f'=============={col[i]}================')
#         for j in range(feature):
#             x_train_data = np.c_[x_train_data, x_train[:,col[i][j]]]
#             x_test_data = np.c_[x_test_data, x_test[:,col[i][j]]]

#         x_train_data = x_train_data[:,1:]
#         x_test_data = x_test_data[:,1:]

#         for i in range(x_test_data.shape[0]):
#             if knn.predict(x_test_data[i], x_train_data, y_label) == y_ans[i]:
#                 score += 1
#         print(f'score:{round(score*100/75,2)}')

