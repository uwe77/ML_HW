# Sequential Forward Selection
import numpy as np
from itertools import combinations_with_replacement
from data import data_space, data

def accuracy(y, y_ans):
    score = 0
    for i in range(len(y)):
        if y[i] == y_ans[i]:
            score += 1
    score = round(score*100/len(y), 2)
    return score

def SFS(obj, d_space):
    f_selected = []
    f_best = []
    ds = d_space.get_k_fold(2)
    x_space = [np.vstack((ds[0][0].get_feature_in_matrix(),
                        ds[0][1].get_feature_in_matrix())),
             np.vstack((ds[1][0].get_feature_in_matrix(),
                        ds[1][1].get_feature_in_matrix()))]
    y_space = [np.hstack(([0 for i in range(len(ds[0][0]))],
                        [1 for i in range(len(ds[0][1]))])),
                np.hstack(([0 for i in range(len(ds[1][0]))],
                        [1 for i in range(len(ds[1][1]))]))]
    f_num = x_space[0].shape[1]

    for row in range(f_num):
        f_selected = combinations_with_replacement(range(f_num), row+1)
        for col in len(f_selected):
            score = 0
            for times in range(2):
                train_data = x_space[1 - times]
                y = y_space[1 - times]
                test_data = x_space[times]
                y_ans = y_space[times]
                obj.fit(train_data[:, f_selected[col]], y)
                y_predict = obj.predict(test_data[:, f_selected[col]])  
                score += accuracy(y_predict, y_ans)
            score = round(score/2, 2)
            if score > best_score:
                best_score = score
                f_best = f_selected[col]