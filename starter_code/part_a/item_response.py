from utils import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    N, M = theta.shape[0], beta.shape[0]
    data = data.toarray()
    C = np.nan_to_num(data, nan=0)
    mask = 1 - np.isnan(data)
    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_1 = np.sum(tmb * C)
    term_2 = np.sum(np.log(1 + np.exp(tmb)) * mask)
    log_lklihood = term_1 - term_2

    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    data = data.toarray()
    C = np.nan_to_num(data, nan=0)
    mask = 1 - np.isnan(data)
    N = theta.shape[0]
    M = beta.shape[0]

    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_2 = sigmoid(tmb) * mask
    theta_cap = -np.sum(C - term_2, axis=1)
    theta -= (lr * theta_cap)

    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_2 = sigmoid(tmb) * mask
    beta_cap = np.sum(C - term_2, axis=0)
    beta -= (lr * beta_cap)

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(data.shape[0])
    beta = np.zeros(data.shape[1])

    val_acc_lst = []
    train_llk = []
    val_llk = []
    val_matrix = dict_to_sparse(val_data, theta.shape[0], beta.shape[0])

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llk.append(-neg_lld)
        val_llk.append(-neg_log_likelihood(val_matrix, theta=theta, beta=beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_llk, val_llk


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def plot_training_curve(train_llk, val_llk):
    plt.plot(train_llk, label="Train")
    plt.plot(val_llk, label="Validation")
    plt.xlabel("Num Iterations")
    plt.ylabel("Log Likelihood")
    plt.legend()
    plt.title("Training curve")
    plt.gcf().set_size_inches([10, 7])
    plt.savefig('plots/irt/training_curve.png')
    plt.show()


def plot_questions(questions, theta, beta):
    theta = np.tile(theta, (beta.shape[0], 1)).T
    pc_ij = sigmoid(theta - beta)[:, questions]
    for i, p in enumerate(pc_ij.T):
        # plt.scatter(theta[:, 0], p, label=f"Question {questions[i]}")
        plt.plot(theta[:, 0], p, label=f"Question {questions[i]}")
    plt.xlabel("Theta")
    plt.ylabel("P(c = 1|Theta, Beta)")
    plt.legend()
    plt.title('Probability of correct response')
    plt.savefig('plots/irt/questions.png')
    plt.show()


def predict(data, theta, beta):
    N, M = theta.shape[0], beta.shape[0]
    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    return sparse_matrix_predictions(data, sigmoid(tmb))


# WEIGHTED VARIANTS

def neg_log_likelihood_weighted(data, weights, theta, beta):
    N, M = theta.shape[0], beta.shape[0]
    data = data.toarray()
    C = np.nan_to_num(data, nan=0)
    mask = 1 - np.isnan(data)
    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_1 = np.sum(tmb * C * weights)
    term_2 = np.sum(np.log(1 + np.exp(tmb)) * mask * weights)
    log_lklihood = term_1 - term_2
    return -log_lklihood


def update_theta_beta_weighted(data, weights, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    data = data.toarray()
    mask = 1 - np.isnan(data)
    C = np.nan_to_num(data, nan=0) * weights
    N = theta.shape[0]
    M = beta.shape[0]

    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_2 = sigmoid(tmb) * mask * weights
    theta_cap = -np.sum(C - term_2, axis=1)
    theta -= (lr * theta_cap)

    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_2 = sigmoid(tmb) * mask * weights
    beta_cap = np.sum(C - term_2, axis=0)
    beta -= (lr * beta_cap)

    return theta, beta


def irt_weighted(data, weights, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(data.shape[0])
    beta = np.zeros(data.shape[1])

    val_acc_lst = []
    train_llk = []
    val_llk = []
    val_matrix, val_weights = dict_to_sparse_weighted(val_data, theta.shape[0],
                                                      beta.shape[0])

    for i in range(iterations):
        neg_lld = neg_log_likelihood_weighted(data, weights, theta=theta,
                                              beta=beta)
        train_llk.append(-neg_lld)
        val_llk.append(
            -neg_log_likelihood_weighted(val_matrix, val_weights, theta=theta,
                                         beta=beta))
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta_weighted(data, weights, lr, theta, beta)

    return theta, beta, val_acc_lst, train_llk, val_llk


def main():
    # train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    lr = 0.01
    iterations = 14

    # For weighted IRT:
    # N = len(set(train_data['user_id']))
    # d = len(set(train_data['question_id']))
    # sparse_matrix, weights = dict_to_sparse_weighted(train_data, N, d)
    # theta, beta, val_acc_list, train_llk, val_llk = \
    #                                                irt_weighted(sparse_matrix,
    #                                                              weights,
    #                                                              val_data, lr,
    #                                                              iterations)

    theta, beta, val_acc_list, train_llk, val_llk = irt(sparse_matrix,
                                                        val_data, lr,
                                                        iterations)
    score = evaluate(test_data, theta=theta, beta=beta)
    print(f'Test Score: {score}')
    plot_training_curve(train_llk, val_llk)
    np.random.seed(0)
    plot_questions([559, 684, 835, 1216, 1653], sorted(theta), beta)

    # For private test:
    # private_test = load_private_test_csv('../data')
    # preds = predict(private_test, theta, beta)
    # private_test['is_correct'] = preds
    # save_private_test_csv(private_test, 'predictions.csv')


if __name__ == "__main__":
    main()
