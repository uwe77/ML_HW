import numpy as np
from data import data_space, data
from lda import LDA
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sfs import SFS
# check which datasets can be imported
# list_available_datasets()

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
# print(d_space)
    

SFS(None, d_space, 2)
# for times in range(2): # times = k-fold
#     train_data_space = ds[times] # data_space
#     test_data_space = ds[1-times] # data_space
#     train_data = np.vstack((train_data_space[0].get_feature_in_matrix(),
#                             train_data_space[1].get_feature_in_matrix())) # festure matrix
#     y = np.hstack(([0 for i in range(len(train_data_space[0]))],
#                     [1 for i in range(len(train_data_space[1]))])) # ans array
#     test_data = np.vstack((test_data_space[0].get_feature_in_matrix(),
#                             test_data_space[1].get_feature_in_matrix())) # festure matrix
#     y_ans = np.hstack(([0 for i in range(len(test_data_space[0]))],
#                         [1 for i in range(len(test_data_space[1]))])) # ans array
#     lda.fit(train_data, y)
#     lda_ans = lda.predict(test_data)
#     score = 0
#     for i in range(len(lda_ans)):
#         if lda_ans[i] == y_ans[i]:
#             score += 1
#     score = round(score*100/len(lda_ans), 2)
# print(score)

# lda.fit(d_space)