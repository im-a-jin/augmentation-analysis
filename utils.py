import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def plot_datapoints(x, y):
    y_ind = ((y + 1) / 2).int()
    x_pos = x[torch.argwhere(y==1).squeeze()]
    x_neg = x[torch.argwhere(y==-1).squeeze()]
    plt.scatter(x_pos[:, 0], x_pos[:, 1], marker='o', c='red',
                edgecolors='black')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], marker='^', c='blue',
                edgecolors='black')
    plt.show()

def plot_history(history, index='loss'):
    epochs = np.arange(1, len(history['train_'+index])+1)
    plt.plot(epochs, history['train_'+index], label='train_'+index)
    plt.plot(epochs, history['test_'+index], label='test_'+index)
    plt.xlabel('epochs')
    plt.ylabel(index)
    plt.legend()
    plt.show()

def plot_sweep(x, histories, xlabel, index='loss'):
    train_y = [h['train_'+index][-1] for h in histories]
    test_y = [h['test_'+index][-1] for h in histories]
    plt.plot(x, train_y, label='train_'+index)
    plt.plot(x, test_y, label='test_'+index)
    plt.xlabel(xlabel)
    plt.ylabel(index)
    plt.legend()
    plt.show()

