from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
from part_a.item_response import dict_to_sparse
import matplotlib.pyplot as plt

from starter_code.utils import load_train_sparse, load_valid_csv, \
    load_public_test_csv


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1|| + ||W^2||.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2)
        h_w_norm = torch.norm(self.h.weight, 2)
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        out_1 = torch.sigmoid(self.g(inputs))
        out = torch.sigmoid(self.h(out_1))
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    val_sparse = dict_to_sparse(valid_data, num_student, train_data.shape[1])
    val_sparse = torch.FloatTensor(val_sparse.toarray())

    train_loss_lst = []
    val_loss_lst = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        val_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]


            loss = torch.sum((output - target) ** 2.) + (
                    (lamb / 2) * model.get_weight_norm())


            loss.backward()

            train_loss += loss.item()
            optimizer.step()

            val_inputs = Variable(val_sparse[user_id]).unsqueeze(0)
            val_target = val_inputs.clone()
            nan_mask_val = np.isnan(val_sparse[user_id].unsqueeze(0).numpy())
            val_target[0][nan_mask_val] = output[0][nan_mask_val]
            val_loss_internal = torch.sum((output - val_target) ** 2.)
            val_loss += val_loss_internal.item()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
    plt.plot(train_loss_lst, label="Train")
    plt.plot(val_loss_lst, label="Validation")
    plt.legend()
    plt.savefig("plots/nn/nn.png")


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    # Set model hyperparameters.
    k_lst = [10, 50, 100, 200, 500]

    # Set optimization hyperparameters.
    config = {
        'k': k_lst[0],
        'lr': 0.01,
        'num_epochs': 80,
        'lamb': 0.001  # 0.001 for regularized version
    }
    print(f'Configuration: {config}')
    k = config['k']
    lr = config['lr']
    num_epochs = config['num_epochs']
    lamb = config['lamb']

    model = AutoEncoder(train_matrix.shape[1], k)
    train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data,
          num_epochs)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f'Test Accuracy: {test_acc}')


if __name__ == "__main__":
    main()
