from utils import *

import numpy as np
import matplotlib.pyplot as plt
import scipy


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
    data = np.nan_to_num(data.toarray(), nan=0)
    # term_1 = theta @ np.sum(data, axis=1)
    # term_2 = beta @ np.sum(data, axis=0)
    # term_3 = 0
    # for i in range(N):
    #     for j in range(M):
    #         term_3 += np.logaddexp(0, theta[i] - beta[j]) * data[i][j]
    # log_lklihood = term_1 + term_2 - term_3
    log_lklihood = np.sum(np.log(
        sigmoid(np.tile(theta, (M, 1)).T - np.tile(beta, (N, 1)))) * data)
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
    # theta_cap = np.zeros_like(theta)
    # beta_cap = np.zeros_like(beta)
    data = np.nan_to_num(data.toarray(), nan=0)
    N = theta.shape[0]
    M = beta.shape[0]
    # print(f'beta_cap: {beta_cap.shape}')
    beta_mat = np.vstack([beta] * N)
    # print(f'beta_mat: {beta_mat.shape}')
    theta_vec = 1 / (1 + np.exp(theta.reshape(-1, 1) - beta_mat))
    # print(f'theta_vec: {theta_vec.shape}')
    theta_vec = theta_vec * data
    theta_cap = -np.sum(theta_vec, axis=1)
    # print(f'theta_cap: {theta_cap.shape}')
    # for i in range(N):
    #     denom = 1 + np.exp(theta[i] - beta)
    #     theta_cap[i] = np.sum(1 / denom)
    theta -= (lr * theta_cap)

    theta_mat = np.vstack([theta] * M)
    # print(f'data: {data.shape}')
    # print(f'theta: {theta.shape}')
    # print(f'beta: {beta.shape}')
    # print(f'theta_mat: {theta_mat.shape}')
    beta_vec = 1 / (1 + np.exp(theta_mat - beta.reshape(-1, 1)))
    # print(f'beta_vec: {beta_vec.shape}')
    beta_vec = beta_vec.T * data
    beta_cap = np.sum(beta_vec, axis=0)
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
    # theta = np.random.randn(data.shape[0])
    # beta = np.random.randn(data.shape[1])
    # theta = -2 * np.ones(data.shape[0])
    # beta = 2 * np.ones(data.shape[1])
    theta = -10 * np.ones(data.shape[0])
    beta = 10 * np.ones(data.shape[1])

    val_acc_lst = []
    train_llk = []
    val_llk = []
    val_matrix = dict_to_sparse(val_data, theta.shape[0], beta.shape[0])

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_llk.append(-neg_lld)
        val_llk.append(-neg_log_likelihood(val_matrix, theta, beta))
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
    plt.show()


def plot_questions(questions, beta):
    theta = np.tile(np.linspace(-5, 5, 100), (beta.shape[0], 1)).T
    pc_ij = sigmoid(theta - beta)[:, questions]
    for i, p in enumerate(pc_ij.T):
        plt.plot(theta[:, 0], p, label=f"Question {questions[i]}")
    plt.xlabel("Theta")
    plt.ylabel("P(c|Theta, Beta)")
    plt.legend()
    plt.show()


def dict_to_sparse(data_dict, N, d):
    mat = scipy.sparse.csc_matrix((N, d))
    for i in range(len(data_dict['user_id'])):
        mat[data_dict['user_id'][i], data_dict['question_id'][i]] = data_dict[
            'is_correct'][i]
    return mat


def main():
    train_data = load_train_csv("../data")
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
    # iterations = 17
    # iterations = 32
    iterations = 100
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
    q = beta.shape[0]
    N = theta.shape[0]
    print(
        f'probabilities: {np.mean(sigmoid(np.vstack([theta] * q) - np.vstack([beta] * N).T))}')
    # print(f'theta: {theta}')
    # print(f'beta: {beta}')
    train_score = evaluate(train_data, theta, beta)
    print(f'Train Score: {train_score}')
    print(f'Test Score: {score}')
    plot_training_curve(train_llk, val_llk)
    np.random.seed(0)
    plot_questions(np.random.choice(sparse_matrix.shape[1], 5), beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
