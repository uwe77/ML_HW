import numpy as np

class LDA:
    def __init__(self):
        self.means_ = {}
        self.priors_ = {}
        self.cov_ = None
        self.weights_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Separate the data into classes
        class_labels = np.unique(y)

        # Calculate means, priors, and covariance
        for label in class_labels:
            X_class = X[y == label]
            self.means_[label] = np.mean(X_class, axis=0)
            self.priors_[label] = X_class.shape[0] / X.shape[0]
        
        self.cov_ = np.cov(X.T, bias=True)  # pooled covariance matrix

        # Calculate the weights and intercept
        inv_cov = np.linalg.inv(self.cov_)
        self.weights_ = np.dot(inv_cov, (self.means_[1] - self.means_[0]))
        self.intercept_ = (
            -0.5 * np.dot(np.dot(self.means_[1], inv_cov), self.means_[1])
            + 0.5 * np.dot(np.dot(self.means_[0], inv_cov), self.means_[0])
            + np.log(self.priors_[1] / self.priors_[0])
        )

    def predict(self, X):
        # Apply the linear discriminant function
        scores = X.dot(self.weights_) + self.intercept_
        return np.where(scores > 0, 1, 0)

    def predict_proba(self, X):
        # Calculate the probability estimates
        scores = X.dot(self.weights_) + self.intercept_
        return 1 / (1 + np.exp(-scores))

# Example usage:
# lda = LDA()
# lda.fit(X_train, y_train)
# y_pred = lda.predict(X_test)
