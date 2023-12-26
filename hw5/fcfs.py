# Fisher’s Criterion
import random
import numpy as np
from data import data_space, data

def FCFS(d_space: data_space, k: int):
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
    
    
    for times in range(k):
        f_means=np.empty((0))
        for i in range(x_space[times].shape[1]):
            f_means = np.append(f_means, np.mean(x_space[times][:,i]))
            f_diff = x_space[times][:,i] - f_means[-1]
            print(f_diff.shape)
        f_mean = np.mean(f_means)
        