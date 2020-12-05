# TODO: complete this file.
from part_a.item_response import *
from sklearn.impute import KNNImputer
from part_a.neural_network import *
from utils import *


def neural_network(train_data, zero_train_data, valid_data):
    pass


def irt(train_data, val_data):
    lr = 0.01
    iterations = 15
    theta, beta, val_acc_list, train_llk, val_llk = irt(train_data, val_data,
                                                        lr, iterations)
    N = theta.shape[0]
    M = beta.shape[0]

    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    predictions = sigmoid(tmb)
    return predictions


def knn(train_matrix):
    nbrs = KNNImputer(n_neighbors=11)
    # We use NaN-Euclidean distance measure.
    return nbrs.fit_transform(train_matrix)


def main():
    # create sample data
    nn_sample = ...
    irt_sample = ...
    knn_sample = ...
    val_data = load_valid_csv("../data")
    zero_train_matrix = nn_sample.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(nn_sample)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    nn_sample = torch.FloatTensor(nn_sample)

    nn_predictions = neural_network(nn_sample, zero_train_matrix, val_data)
    irt_predictions = irt(irt_sample, val_data)
    knn_predictions = knn(knn_sample)

    model = (nn_predictions + irt_predictions + knn_predictions)/3




if __name__ == '__main__':
    main()
