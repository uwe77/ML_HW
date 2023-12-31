import numpy as np
from data import data_space

def fs(d_space:data_space):
    
    m0 = np.mean(d_space[0].get_feature_in_matrix(), axis=0)
    m1 = np.mean(d_space[1].get_feature_in_matrix(), axis=0)
    m = np.mean(np.vstack((m0, m1)), axis=0)
    sw0 = sum((d_space[0].get_feature_in_matrix() - m0)**2)
    sw1 = sum((d_space[1].get_feature_in_matrix() - m1)**2)
    n0 = len(d_space[0].get_feature_in_matrix())
    n1 = len(d_space[1].get_feature_in_matrix())
    n = n0 + n1
    sb0 = n0/n * (m0 - m)**2
    sb1 = n1/n * (m1 - m)**2
    sb = sb0 + sb1
    sw = (sw0 + sw1)/n
    score = []
    for i in range(len(sw)):
        if sw[i] == 0:
            score.append(0)
        else:
            score.append(sb[i]/sw[i])
    score_index = np.argsort(score)[::-1]
    return score_index, score