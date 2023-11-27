from data import *
from SVM import RBF_SVM as rbf
import re
import numpy as np
import matplotlib.pyplot as plt


d_space = data_space()

f = open('iris.txt', 'r')
for line in f.readlines():
    s = re.split(r'\s+', line)
    data_input = data(int(s[4]))
    data_input[:] = [float(i) for i in s[:4]]
    d_space.append_data(data_input)
f.close()

combination = [[0,0,1], [1,0,2], [2,1,2]]
C = [1, 5, 10, 50, 100, 500, 1000]
sigma = range(-100, 105, 5)
cv_values = np.zeros((len(sigma), len(C)))
ds = d_space.get_k_fold(2)

for c in range(len(C)):
    for s in range(len(sigma)):
        for times in range(2):
            train_data_space = ds[times]
            test_data_space = ds[1-times]
            test_data = np.vstack((test_data_space[0].get_feature_in_matrix(),
                                    test_data_space[1].get_feature_in_matrix(),
                                    test_data_space[2].get_feature_in_matrix()))
            y_ans = np.hstack(([1 for i in range(len(test_data_space[0]))], 
                                [2 for i in range(len(test_data_space[1]))],
                                [3 for i in range(len(test_data_space[2]))]))
            svm = [rbf(C=C[c], sigama=1.05**sigma[s]) for i in range(3)]
            for models in combination:
                train_data = np.vstack((train_data_space[models[1]].get_feature_in_matrix(),
                                        train_data_space[models[2]].get_feature_in_matrix()))
                y_label = np.hstack(([1 for i in range(len(train_data_space[models[1]]))], 
                                     [-1 for i in range(len(train_data_space[models[2]]))]))
                svm[models[0]].fit(train_data, y_label)
            svm_ans = []
            for i in range(3):
                tmp_ans = []
                for sa in svm[combination[i][0]].predict(test_data):
                    if sa == 1:
                        tmp_ans.append(combination[i][1]+1)
                    else:
                        tmp_ans.append(combination[i][2]+1)
                svm_ans.append(tmp_ans)
            svm_ans = np.array(svm_ans)
            svm_ans = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=svm_ans)
            score = 0
            for i in range(len(svm_ans)):
                if svm_ans[i] == y_ans[i]:
                    score += 1
            score = round(score*100/len(svm_ans), 2)
            cv_values[s, c] += score
        cv_values[s, c] = round(cv_values[s,c]/2, 2)


# Create a new figure with a specified size (width, height)
plt.figure(figsize=(10,10))

plt.imshow(cv_values, cmap='cool', interpolation='nearest')

# Add text annotations for each block
for i in range(cv_values.shape[0]):
    for j in range(cv_values.shape[1]):
        plt.text(j, i, round(cv_values[i, j], 2), ha='center', va='center', color='black')

# Label rows and columns
row_labels = sigma
col_labels = C
plt.xticks(np.arange(cv_values.shape[1]), col_labels)
plt.yticks(np.arange(cv_values.shape[0]), row_labels)
plt.axis('tight')

# Add row and column titles
plt.xlabel('C')
plt.ylabel('sigama')

plt.colorbar(label='Cross-Validation')
plt.savefig('images/cv.png', dpi=300)