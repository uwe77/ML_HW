import re
import numpy as np
import matplotlib.pyplot as plt
class KNN:
    def __init__(self, K) -> None:
        self._k = K

    def _distant(self, n1, n2):
        return ((n1 - n2)**2)**0.5
    
    def predict(self, x, y, x_test):
        distances = [self._distant(x_test, i) for i in x]
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

k1 = KNN(1)
k3 = KNN(3)

x_input = np.array(x_input)
for i in range(4):
    for j in range(i+1, 4):
        plt.scatter(x_input[:,i], x_input[:,j])
        # plt.show()
        plt.savefig(f'feature{i}_{j}.png')
        plt.clf()

x_test = x_input[75:]
x_train = x_input[:75]

