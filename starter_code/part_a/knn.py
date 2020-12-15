from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("User Based Validation Accuracy: {}".format(acc))
    return acc



def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Item Based Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_list = [1, 6, 11, 16, 21, 26]
    user_accuracies = []
    item_accuracies = []
    for k in k_list:
        print(f"k = {k}: ")
        user_accuracies.append(knn_impute_by_user(sparse_matrix, val_data, k))
        item_accuracies.append(knn_impute_by_item(sparse_matrix, val_data, k))
    k_user = int(np.argmax(user_accuracies))
    k_item = int(np.argmax(item_accuracies))
    print("Test for k*:")
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, k_list[k_user])
    test_acc_item = knn_impute_by_item(sparse_matrix, test_data, k_list[k_item])
    plt.plot(k_list, user_accuracies)
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("User-Based Collaborative Filtering")
    plt.savefig('plots/knn/user.png')
    plt.show()

    plt.plot(k_list, item_accuracies)
    plt.xlabel("k")
    plt.ylabel("Validation Accuracy")
    plt.title("Item-Based Collaborative Filtering")
    plt.savefig('plots/knn/item.png')
    plt.show()

    print("\n")
    print("Summary")
    print("----------------------")
    print("User based")
    print("---------------")
    print(f"k selected: {k_list[k_user]}")
    print(f"test acc: {test_acc_user}\n")
    print("Item based")
    print("---------------")
    print(f"k selected: {k_list[k_item]}")
    print(f"test acc: {test_acc_item}\n")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
