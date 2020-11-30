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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    N, M = theta.shape[0], beta.shape[0]
    data = data.toarray()
    C = np.nan_to_num(data, nan=0)
    mask = 1 - np.isnan(data)
    tmb = np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1))
    term_1 = np.sum(tmb * C)
    term_2 = np.sum(np.log(1 + np.exp(tmb)) * mask)
    log_lklihood = term_1 - term_2
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
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
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
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
    # TODO: Initialize theta and beta.
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
    plt.savefig('plots/irt/training_curve.png')
    plt.show()


def plot_questions(questions, beta):
    theta = np.tile(np.linspace(-5, 5, 100), (beta.shape[0], 1)).T
    pc_ij = sigmoid(theta - beta)[:, questions]
    for i, p in enumerate(pc_ij.T):
        plt.plot(theta[:, 0], p, label=f"Question {questions[i]}")
    plt.xlabel("Theta")
    plt.ylabel("P(c|Theta, Beta)")
    plt.legend()
    plt.savefig('plots/irt/questions.png')
    plt.show()


def dict_to_sparse(data_dict, N, d):
    mat = sparse.lil_matrix((N, d))
    for i in range(len(data_dict['user_id'])):
        mat[data_dict['user_id'][i], data_dict['question_id'][i]] = data_dict[
            'is_correct'][i]
    return mat


def main():
    # train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    iterations = 20
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    theta, beta, val_acc_list, train_llk, val_llk = irt(sparse_matrix, val_data,
                                                        lr, iterations)
    score = evaluate(test_data, theta=theta, beta=beta)
    print(f'Test Score: {score}')
    plot_training_curve(train_llk, val_llk)
    np.random.seed(0)
    plot_questions(np.random.choice(sparse_matrix.shape[1], 5), beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
