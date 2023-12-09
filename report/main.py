from SVM import RBF_SVM as rbf
from optimizers.gwo import GWO as gwo
import numpy as np
import re
import matplotlib.pyplot as plt
from data import *


d_space = data_space()
f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(int(s[4]))
    data_input[:] = [float(i) for i in s[:4]]
    d_space.append_data(data_input)
f.close()
ds = d_space.get_k_fold(2) # list(data_space)

def fitness(x):
    cv = 0
    features = [i>=1 for i in x]
    # print(features)
    for i in range(2):
        train_data_space = ds[i] # data_space
        test_data_space = ds[1-i] # data_space
        test_data = np.vstack((test_data_space[0].get_feature_in_matrix(),
                                test_data_space[1].get_feature_in_matrix())) # festure matrix
        y_ans = np.hstack(([1 for i in range(len(test_data_space[0]))], 
                            [2 for i in range(len(test_data_space[1]))])) # ans array
        train_data = np.vstack((train_data_space[0].get_feature_in_matrix(),
                                train_data_space[1].get_feature_in_matrix()))
        y_label = np.hstack(([1 for i in range(len(train_data_space[0]))], 
                                [-1 for i in range(len(train_data_space[1]))]))
        test_data = np.delete(test_data, np.where(features==False), axis=1)
        # print(test_data)
        train_data = np.delete(train_data, np.where(features==False), axis=1)
        # for i in range(len(features)):
        #     if not features[i]:
        #         train_data = np.delete(train_data, i, axis=1)
        #         test_data = np.delete(test_data, i, axis=1)
        rbf_svm = rbf(C=50, sigama=1.05**-50)
        rbf_svm.fit(train_data, y_label)
        tmp_ans = rbf_svm.predict(test_data)
        score = 0
        for j in range(len(tmp_ans)):
            # print(tmp_ans[j], y_ans[j])
            if tmp_ans[j] ==1 and y_ans[j] == 1:
                score += 1
            elif tmp_ans[j] == -1 and y_ans[j] == 2:
                score += 1
        cv += score/len(tmp_ans)
    # print(round(cv/2 ,2))
    return cv/2

print(fitness(gwo(fitness=fitness, max_iter=20, n=10, dim=4, minx=0, maxx=2)))
# print(gwo(fitness=fitness, max_iter=20, n=10, dim=4, minx=0, maxx=2))