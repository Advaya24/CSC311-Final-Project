# TODO: complete this file.
from part_a.item_response import *
from sklearn.impute import KNNImputer
from part_a.neural_network import *
from utils import *


def irt_ensemble(train_data, train_weights, val_data):
    lr = 0.01
    iterations = 15
    theta, beta, val_acc_list, train_llk, val_llk = irt_weighted(train_data,
                                                                 train_weights,
                                                                 val_data,
                                                                 lr, iterations)
    N = theta.shape[0]
    M = beta.shape[0]

    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    predictions = sigmoid(tmb)
    return predictions


def knn_user(train_matrix):
    nbrs = KNNImputer(n_neighbors=11)
    # We use NaN-Euclidean distance measure.
    return nbrs.fit_transform(train_matrix)


def knn_item(train_matrix):
    nbrs = KNNImputer(n_neighbors=21)
    # We use NaN-Euclidean distance measure.
    return nbrs.fit_transform(train_matrix.T).T


def main():
    # create sample data
    train_full = load_train_csv('../data')
    test_data = load_public_test_csv('../data')
    private_test = load_private_test_csv('../data')
    n = len(train_full['user_id'])
    N = len(set(train_full['user_id']))
    d = len(set(train_full['question_id']))
    irt1_idx = np.random.choice(n, n)
    irt2_idx = np.random.choice(n, n)
    irt3_idx = np.random.choice(n, n)
    irt1_dict = sample_from_dict(train_full.copy(), irt1_idx)
    irt2_dict = sample_from_dict(train_full.copy(), irt2_idx)
    irt3_dict = sample_from_dict(train_full.copy(), irt3_idx)
    irt1_sample, irt1_weights = dict_to_sparse_weighted(irt1_dict, N, d)
    irt2_sample, irt2_weights = dict_to_sparse_weighted(irt2_dict, N, d)
    irt3_sample, irt3_weights = dict_to_sparse_weighted(irt3_dict, N, d)
    val_data = load_valid_csv("../data")

    # user = user_sample.toarray()
    # user[user == 0] = -1
    # user *= user_weights
    # item = item_sample.toarray()
    # item[item == 0] = -1
    # item *= item_weights
    print('First dataset: ')
    irt1_predictions = irt_ensemble(irt1_sample, irt1_weights, val_data)
    print('\nSecond dataset: ')
    irt2_predictions = irt_ensemble(irt2_sample, irt2_weights, val_data)
    print('\nThird dataset: ')
    irt3_predictions = irt_ensemble(irt3_sample, irt3_weights, val_data)
    # user_out = np.nan_to_num(knn_user(user), nan=0)
    # item_out = np.nan_to_num(knn_item(item), nan=0)
    # user_predictions = 0.5 + (user_out / 2)
    # item_predictions = 0.5 + (item_out / 2)

    # if user_predictions.shape != irt_predictions.shape:
    #     missing_col = list(
    #         set(range(d)).difference(set(user_dict['question_id'])))
    #     missing_col.append(d)
    #     user_inter = np.ones((N, d)) * 0.5
    #     for i in range(1, len(missing_col)-1):

    predictions = (irt1_predictions + irt2_predictions + irt3_predictions) / 3
    val_acc = sparse_matrix_evaluate(val_data, predictions)
    test_acc = sparse_matrix_evaluate(test_data, predictions)
    # private_test['is_correct'] = sparse_matrix_predictions(private_test,
    #                                                        predictions)
    # save_private_test_csv(private_test, 'predictions.csv')
    print("\nFinal Accuracies: ")
    print("------------------------------")
    print(f'Val accuracy: {val_acc}')
    print(f'Test accuracy: {test_acc}')


if __name__ == '__main__':
    main()
