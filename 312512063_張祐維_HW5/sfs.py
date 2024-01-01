# Sequential Forward Selection
import numpy as np
from data import data_space, data

def accuracy(y, y_ans):
    score = 0
    for i in range(len(y)):
        if y[i] == y_ans[i]:
            score += 1
    score = round(score*100/len(y), 2)
    return score

def SFS(OBJ, d_space, k):
    ds = d_space.get_k_fold(k)
    x_space = []
    y_space = []
    for i in range(k):
        x_tmp = np.empty((0, len(ds[i][0].get_feature_in_matrix()[0])))
        y_tmp = np.empty((0))
        for j in range(len(ds[i])):
            x_tmp = np.vstack((x_tmp, ds[i][j].get_feature_in_matrix()))
            y_tmp = np.hstack((y_tmp, [j for k in range(len(ds[i][j]))]))
        x_space.append(x_tmp)
        y_space.append(y_tmp)
    best_score_list = []
    best_f_list = []
    sample = range(x_space[0].shape[1])
    for row in range(x_space[0].shape[1]):
        best_score_list.append(0)
        f_selecte_list = []
        for i in range(len(sample)):
            if best_f_list:
                if i not in best_f_list:
                    f_tmp = []
                    for j in best_f_list:
                        f_tmp.append(j)
                    f_tmp.append(i)
                    f_selecte_list.append(f_tmp)
            else:
                f_selecte_list.append([i])
        for col in f_selecte_list:
            score = 0
            for times in range(k):
                train_data = np.empty((0, x_space[0].shape[1]))
                y = np.empty((0))
                for i in range(k):
                    if i != times:
                        train_data = np.vstack((train_data, x_space[i]))
                        y = np.hstack((y, y_space[i]))
                test_data = x_space[times]
                mask = np.isin(range(x_space[0].shape[1]), col)
                train_data = train_data[:, mask]
                test_data = test_data[:, mask]
                y_ans = y_space[times]
                obj = OBJ()
                obj.fit(train_data, y)
                y_predict = obj.predict(test_data)
                score += accuracy(y_predict, y_ans)
            score = round(score/k, 2)
            if score > best_score_list[-1]:
                best_score_list.pop()
                best_score_list.append(score)
                best_f_list = col.copy()
    return best_f_list, best_score_list