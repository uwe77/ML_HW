# import numpy as np

# # Sample dataset
# X = np.array([[1, 2], [2, 3], [3, 4], [5, 6], [6, 7]])
# y = np.array([0, 0, 1, 1, 1])

# # Function to calculate Euclidean distance between two points
# def euclidean_distance(x1, x2):
#     return np.sqrt(np.sum((x1 - x2) ** 2))

# # k-NN classifier function
# def knn_predict(X_train, y_train, x_test, k):
#     distances = [euclidean_distance(x_test, x) for x in X_train]
#     k_indices = np.argsort(distances)[:k]
#     k_nearest_labels = [y_train[i] for i in k_indices]
#     most_common = np.bincount(k_nearest_labels).argmax()
#     return most_common

# # Test data
# x_test = np.array([4, 5])
# k = 3

# # Predict the class label
# predicted_label = knn_predict(X, y, x_test, k)
# print(f"Predicted Label: {predicted_label}")
