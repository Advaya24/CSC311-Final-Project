from sklearn.impute import KNNImputer
import numpy as np
from scipy.spatial.distance import hamming


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

    mat_euclidean = nbrs_euclidean.fit_transform(train_data.T).T
    mat_hamming = nbrs_hamming.fit_transform(train_data.T).T

    print(f'train: \n{train_data}')
    print(f'nan_euclidean: \n{mat_euclidean}')
    print(f'accuracy: {np.mean(mat_euclidean == full_data)}')
    print(f'hamming: \n{mat_hamming}')
    print(f'accuracy: {np.mean(mat_hamming == full_data)}')
    print(f'actual: \n{full_data}')
