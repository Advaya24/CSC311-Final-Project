from sklearn.impute import KNNImputer
import numpy as np
from scipy.spatial.distance import hamming

'''
This is the code for running the experiment demonstrated in section 3 of Part B
'''


def pairwise_callable(X, Y, **kwds):
    return hamming(X, Y)


if __name__ == '__main__':
    train_data = np.array(
        [[1, 1, 1], [1, np.nan, 1], [np.nan, 1, 0], [1, np.nan, 1],
         [1, np.nan, 0]])
    full_data = np.array(
        [[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 1],
         [1, 1, 0]])

    nbrs_euclidean = KNNImputer(n_neighbors=1)
    nbrs_hamming = KNNImputer(n_neighbors=1, metric=pairwise_callable)

    nan_euclidean_imputed = nbrs_euclidean.fit_transform(train_data.T).T
    hamming_imputed = nbrs_hamming.fit_transform(train_data.T).T

    id_test = np.where(np.isnan(train_data))
    print(f'train: \n{train_data}')
    print(f'\nnan_euclidean_imputed: \n{nan_euclidean_imputed}')
    print(f'test accuracy:'
          f' {np.mean(nan_euclidean_imputed[id_test] == full_data[id_test])}')
    print(f'\nhamming_imputed: \n{hamming_imputed}')
    print(f'test accuracy: '
          f'{np.mean(hamming_imputed[id_test] == full_data[id_test])}')
    print(f'\nactual: \n{full_data}')
