import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from parameter import Parameters, parse_parameters
from dataset import XORMixture
from model import NonlinearModel
import trainer
import utils

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=10000)
        
PARAMETERS = {
    'd': 128,
    'm': 500,
    'n': 5000,
    'noise_rate': 0.15,
    'lr': 5.0,
    'epochs': 500,
    'p': 0.0,
    'cluster_var': 1/25,
    'init_var': 1/16,
    'mu': None,
    'seed': 0,
    'plt_idx': [0, -1],
    'test_all': False,
}


def main():
    PARAMS, SWEEPS = parse_parameters(sys.argv, PARAMETERS)

    utils.set_random_seeds(PARAMS.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = XORMixture(
            dims=PARAMS.d,
            var=PARAMS.cluster_var*1/PARAMS.d,
            n=PARAMS.n,
            noise_rate=PARAMS.noise_rate,
            mu=PARAMS.mu,
    )

#   utils.plot_datapoints(dataset.x, dataset.y)

    print(PARAMS, SWEEPS)
    models, histories, labels = [], [], []
    for params in PARAMS:
#       dataset._shuffle()
#       if type(PARAMS[PARAMS.sweep]) is float:
#           PARAMS[PARAMS.sweep] = np.around(PARAMS[PARAMS.sweep], 6)
        print(params)

        transforms = nn.Dropout(p=params.p)
        model = NonlinearModel(
                in_dim=params.d,
                hidden_dim=params.m,
                out_dim=1,
                init_var=params.init_var*1/(params.m*params.d),
        )
        model.to(device)
        m, h = trainer.train(model, dataset, transforms, params.epochs,
                params.lr, params.test_all, device)
        models.append(m)
        histories.append(h)
#       labels.append(PARAMS.sweep + "=" + str(PARAMS[PARAMS.sweep]))

    utils.plot_sweep(PARAMS.sweep_range(), histories, xlabel=PARAMS.sweep,
            kind='loss')
    utils.plot_sweep(PARAMS.sweep_range(), histories, xlabel=PARAMS.sweep,
            kind='acc')
    utils.plot_history(histories, labels, PARAMS.plt_idx, kind='loss')
    utils.plot_history(histories, labels, PARAMS.plt_idx, kind='acc')
#   utils.plot_datapoints(dataset.x, torch.sign(m(dataset.x).squeeze()))

if __name__ == "__main__":
    main()
