import numpy as np
from data import data_space, data
from lda import LDA
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sfs import SFS
import matplotlib.pyplot as plt


# import dataset
breast_cancer = fetch_ucirepo(id=17)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = np.array(breast_cancer.data.features)
Y = np.array(breast_cancer.data.targets).flatten()
d_space = data_space()
for i in range(len(Y)):
    y_trans = lambda j: 0 if Y[j] == 'M' else 1 # M = 0, B = 1
    data_input = data(y_trans(i)) # create data object
    data_input[:] = X[i] # add features
    d_space.append_data(data_input) # add data object to data_space

sfs_fs, sfs_sc = SFS(LDA, d_space, 2) # 2-fold cross validation LDA SFS

plt.figure(figsize=(10,10))
plt.plot(np.arange(len(sfs_sc)), sfs_sc)
plt.xlabel('Cumulatively used features')
plt.ylabel('2-fold CV')
plt.xticks(np.arange(len(sfs_sc)), sfs_fs)
plt.yticks(sfs_sc, sfs_sc)
plt.savefig('images/sfs_cv.png', dpi=300)