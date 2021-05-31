import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score, \
    accuracy_score
import math


def get_neigh_index(filename):
    neigh = np.loadtxt(filename, delimiter=',')
    neigh_index = []
    for i in range(len(neigh)):
        list_index = []
        for j in range(len(neigh[0])):
            if neigh[i][j] == 1:
                list_index.append(j)
        neigh_index.append(list_index)
    neigh_index = torch.tensor(neigh_index)
    return neigh_index


def prepare_data(data, len_recent_time):
    data_recent = []
    for i in range(len(data) - len_recent_time):
        data_recent.append(data[i:i + len_recent_time])
    data_recent = torch.tensor(np.array(data_recent), dtype=torch.float32)
    return data_recent


def loss_function(pred, y, dy_diff, a_dy=6, lambda_=0.005, epsilon=0.9, alpha=0.25, gamma=2):
    zeros = torch.zeros_like(pred, dtype=pred.dtype, device=pred.device)
    focal_loss = -alpha * (1 - pred) ** gamma * y * torch.log(pred) - (1 - alpha) * pred ** gamma * (1 - y) * torch.log(
        1 - pred)
    focal_loss = torch.where(pred * y + (1 - pred) * (1 - y) > epsilon, zeros, focal_loss)
    focal_loss = torch.mean(focal_loss)
    dy_loss_function = nn.L1Loss()
    dy_loss = dy_loss_function(dy_diff, torch.zeros_like(dy_diff, dtype=dy_diff.dtype, device=dy_diff.device))
    dy_loss = a_dy - torch.mean(dy_loss)
    dy_loss = torch.max(torch.tensor(0, device=dy_diff.device, dtype=dy_diff.dtype), dy_loss)

    loss = focal_loss + lambda_ * dy_loss
    return loss, focal_loss, dy_loss


def compute_loss(x, thre_nc, y_dy, y, model, batch_size):
    batch_val = math.ceil(len(x) / batch_size)
    loss_mean = []
    for i in range(batch_val):
        y_pred, y_dy, dy_diff = model(x[i * batch_size:(i + 1) * batch_size],
                                      thre_nc[i * batch_size:(i + 1) * batch_size], y_dy)
        loss, focal_loss, dy_loss = loss_function(y_pred, y[i * batch_size:(i + 1) * batch_size], dy_diff)
        loss_mean.append(loss)
    return tf.cast(np.array(loss_mean).mean(), dtype=tf.float32)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
                            early_stop = EarlyStopping(patience=10,delta=0.000001)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, current_val_loss):
        current_score = current_val_loss

        if self.best_score is None:
            self.best_score = current_score

        elif current_score > self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            print(f'EarlyStopping update val_loss: {self.best_score} --> {current_score}')
            self.best_score = current_score
            self.counter = 0
