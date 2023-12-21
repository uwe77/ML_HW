import numpy as np
from lda import LDA
from ucimlrepo import fetch_ucirepo, list_available_datasets

# check which datasets can be imported
list_available_datasets()

# import dataset
breast_cancer = fetch_ucirepo(id=17)
# alternatively: fetch_ucirepo(name='Heart Disease')

# access data
X = breast_cancer.data.features
y = breast_cancer.data.targets
# sklearn.linear_model.LinearRegression().fit(X, y)

# access metadata
print(breast_cancer.metadata.uci_id)
print(breast_cancer.metadata.num_instances)
print(breast_cancer.metadata.additional_info.summary)

# access variable info in tabular format
print(breast_cancer.variables)
# Example usage:
lda = LDA()
print(f'X:\n{np.array(X).shape}')
print(f'y:\n{np.array(y).flatten()}')
# lda.fit(np.array(X), np.array(y).flatten())
# y_pred = lda.predict(X)