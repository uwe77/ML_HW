import numpy as np
import matplotlib.pyplot as plt
from data import data_space, data  # Ensure these are defined or imported correctly

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
            x_m[:, i] = x_m[:, i] - mean_c[c, i]
        sw += np.dot(x_m.T, x_m) / x_m.shape[0]
        c_a = mean_c[c] - mean_a
        sb += np.dot(c_a.reshape(-1, 1), c_a.reshape(1, -1))

    # Solving the generalized eigenvalue problem for sw^-1 * sb
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(sw).dot(sb))

    # Sorting the eigenvectors by decreasing eigenvalues
    eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]
    # eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

    # Selecting the eigenvector with the largest eigenvalue
    
    # w = eiglist[0][1]
    # print(eiglist[0])
    for i in range(len(eiglist)):
        w = eiglist[i][1]
        # Projecting data onto the selected eigenvector
        projected_data = []
        labels = []
        for c in range(len(d_space)):
            proj = d_space[c].get_feature_in_matrix().dot(w)
            projected_data.append(proj)
            labels += [c] * proj.shape[0]

        # Visualization
        plt.figure(figsize=(10, 6))
        for c, proj in enumerate(projected_data):
            plt.scatter(proj, np.zeros_like(proj), label=f'Class {c}')
        plt.title('Projection of data onto the Fisher\'s Linear Discriminant')
        plt.xlabel('Fisher\'s Linear Discriminant')
        plt.legend()
        image_path = f"./images/fs_feature_{i+1}.png"
        plt.savefig(image_path)
        plt.close()
    sorted_indices = np.argsort(abs(eigvals))[::-1]
    return sorted_indices, abs(eigvals)