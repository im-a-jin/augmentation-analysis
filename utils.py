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

def plot_history(history, labels, indices, kind='loss'):
    epochs = np.arange(1, len(history[0]['train_'+kind])+1)
    ax = plt.gca()
    for i in indices:
        c=next(ax._get_lines.prop_cycler)['color']
        label_str = kind + " @ " + labels[i]
        plt.plot(epochs, history[i]['train_'+kind], label='train_'+label_str,
                color=c, linestyle=':')
        plt.plot(epochs, history[i]['test_'+kind], label='test_'+label_str,
                color=c, linestyle='-')
    plt.xlabel('epochs')
    plt.ylabel(kind)
    plt.legend()
    plt.show()

def plot_sweep(x, histories, xlabel, kind='loss'):
    train_y = [h['train_'+kind][-1] for h in histories]
    test_y = [h['test_'+kind][-1] for h in histories]
    plt.plot(x, train_y, label='train_'+kind)
    plt.plot(x, test_y, label='test_'+kind)
    plt.xlabel(xlabel)
    plt.ylabel(kind)
    plt.legend()
    plt.show()

