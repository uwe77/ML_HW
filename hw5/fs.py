# Fisher’s Criterion
import random
import numpy as np
from data import data_space, data

def FS(d_space: data_space):
    
    sw = 0
    sb = 0
    mean_c = np.empty((0, d_space[0].get_feature_in_matrix().shape[1]))
    for c in range(len(d_space)):
        mean_c = np.vstack((mean_c, np.mean(d_space[c].get_feature_in_matrix(), axis=0)))
    mean_a = np.mean(mean_c, axis=0)

    for c in range(len(d_space)):
        x_m = d_space[c].get_feature_in_matrix().copy()
        for i in range(x_m.shape[1]):
            x_m[:,i] = x_m[:,i] - mean_c[c, i]
        sw += np.dot(x_m.T, x_m) / x_m.shape[0]
        c_a = np.zeros(len(mean_a))
        for i in range(len(mean_a)):
            c_a[i] = mean_c[c,i] - mean_a[i]
        c_a = np.vstack((c_a, np.zeros(c_a.shape[0])))
        sb += np.dot(c_a.T, c_a)
    # sw = np.linalg.eig(sw)[0]
    # sb = np.linalg.eig(sb)[0]
    score = np.zeros(len(sw))
    for i in range(len(sw)):
        score[i] = sb[i,i]/sw[i,i]
        print(score[i])
    return score