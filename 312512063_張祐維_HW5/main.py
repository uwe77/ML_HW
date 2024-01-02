import numpy as np
from data import data_space, data
from lda import LDA
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sfs import SFS, accuracy
from fs import FS
import matplotlib.pyplot as plt


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


##################################SFS+LDA##################################
sfs_fs, sfs_sc = SFS(LDA, d_space, 2) # 2-fold cross validation LDA SFS
max_index = np.argmax(sfs_sc)

plt.figure(figsize=(10,10))
plt.plot(np.arange(len(sfs_sc)), sfs_sc)
plt.scatter(max_index, sfs_sc[max_index], color='red')
plt.axvline(x=max_index, color='green', linestyle='--')
plt.axhline(y=sfs_sc[max_index], color='green', linestyle='--')
plt.xlabel('Cumulatively used features')
plt.ylabel('2-fold CV')
plt.xticks(np.arange(len(sfs_sc)), sfs_fs)
plt.yticks(sfs_sc, sfs_sc)
plt.savefig('images/sfs_cv2.png', dpi=300)
############################################################################

##################################FS+LDA####################################
fs_index, fs_score = FS(d_space)
ds = d_space.get_k_fold(2)
fs_index_list = []
fs_score_list = []
lda_score_list = []
fs_index_list.append(fs_index)
fs_score_list.append(fs_score)
for i in range(len(ds)):
    fs_index_list.append(FS(ds[i])[0])
    fs_score_list.append(FS(ds[i])[1])
    
x_space = []
y_space = []
for i in range(2):
    x_tmp = np.empty((0, len(ds[i][0].get_feature_in_matrix()[0])))
    y_tmp = np.empty((0))
    for j in range(len(ds[i])):
        x_tmp = np.vstack((x_tmp, ds[i][j].get_feature_in_matrix()))
        y_tmp = np.hstack((y_tmp, [j for k in range(len(ds[i][j]))]))
    x_space.append(x_tmp)
    y_space.append(y_tmp)

for fs_times in range(len(fs_index_list)):
    score_list = []
    for col in range(len(fs_index_list[fs_times])):
        score = 0
        for times in range(2):
            train_data = np.empty((0, x_space[0].shape[1]))
            y = np.empty((0))
            for i in range(2):
                if i != times:
                    train_data = np.vstack((train_data, x_space[i]))
                    y = np.hstack((y, y_space[i]))
            test_data = x_space[times]
            mask = np.isin(range(x_space[0].shape[1]), fs_index_list[fs_times][:col+1])
            train_data = train_data[:, mask]
            test_data = test_data[:, mask]
            y_ans = y_space[times]
            obj = LDA()
            obj.fit(train_data, y)
            y_predict = obj.predict(test_data)
            score += accuracy(y_predict, y_ans)
        score = round(score/2, 2)
        score_list.append(score)
    plt.figure(figsize=(20,10))
    plt.plot(np.arange(len(fs_score_list[fs_times])), np.sort(fs_score_list[fs_times])[::-1])
    plt.axvline(x=np.argmax(np.array(score_list)), color='green', linestyle='--')
    plt.axhline(y=np.sort(fs_score_list[fs_times])[::-1][np.argmax(np.array(score_list))], color='green', linestyle='--')
    for i in range(len(score_list)):
        plt.scatter(i, np.sort(fs_score_list[fs_times])[::-1][i], color='red')
        if i == np.argmax(np.array(score_list)):
            plt.text(i, np.sort(fs_score_list[fs_times])[::-1][i], f'{score_list[i]}', ha='right', color='red')
        else:
            plt.text(i, np.sort(fs_score_list[fs_times])[::-1][i], f'{score_list[i]}', ha='right', color='green')  # add text
    plt.xlabel('feature / Cumulatively used features')
    plt.ylabel('Fisher\'s Score / 2-fold CV')
    plt.xticks(np.arange(len(fs_score_list[fs_times])), fs_index_list[fs_times])
    if fs_times == 0:
        plt.title(f'fs with all data')
        plt.savefig(f'images/fs_all.png', dpi=300)
    else:
        plt.title(f'fs with dataset{fs_times}')
        plt.savefig(f'images/fs_k-fold:{fs_times}.png', dpi=300)
    plt.close()

############################################################################