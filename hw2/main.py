import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations as C
class data:
    def __init__(self, y) -> None:
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
        return f'y:{self._label}, feature:{self._feature}'

d1 = data(1)
d1.append(i for i in [1,2,3,4])
print(d1)
print(f'd1.0:{d1[0]}')
for i in d1[:]:
    print(i)

# y = []
# x_input = []
# f = open('iris.txt', 'r')
# for line in f.readlines():
#     s = re.split(r'\s+', line)
#     temp_list = []
#     for i in range(4):
#         temp_list.append(float(s[i]))
#     y.append(int(s[4]))
#     x_input.append(temp_list)
# y = np.array(y)
# x_input = np.array(x_input)