import os
import sys

from absl import app, flags
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import XORMixture
from model import NonlinearModel
import trainer
import utils

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=10000)

FLAGS = flags.FLAGS

flags.DEFINE_integer('d', 128, 'input dimensions')
flags.DEFINE_integer('m', 50, 'hidden dimensions')
flags.DEFINE_integer('n', 5000, 'number of samples')

flags.DEFINE_float('noise_rate', 0.15, 'probability of false labeling')

flags.DEFINE_float('lr', 0.05, 'gradient descent learning rate')
flags.DEFINE_integer('epochs', 300, 'number of training epochs') 
flags.DEFINE_integer('batch_size', 0, 'batch size')
flags.DEFINE_float('p', 0.5, 'dropout probability')

flags.DEFINE_integer('seed', 0, 'random seed')

def main(argv):
    utils.set_random_seeds(FLAGS.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = XORMixture(
            dims=FLAGS.d,
            var=1/(25*FLAGS.d),
            n=FLAGS.n,
            noise_rate=FLAGS.noise_rate,
#           mu=[0.0, 1.0],
    )

#   utils.plot_datapoints(dataset.x, dataset.y)

    models, histories = [], []
    for p in np.arange(0.0, FLAGS.p, 0.1):
        dataset._shuffle()
        transforms = nn.Dropout(p=p)
        model = NonlinearModel(
                in_dim=FLAGS.d,
                hidden_dim=FLAGS.m,
                out_dim=1,
                init_var=1/(16*FLAGS.m*FLAGS.d),
        )
        model.to(device)
        m, h = trainer.train(model, dataset, transforms, FLAGS.lr, FLAGS.epochs,
            device)
        models.append(m)
        histories.append(h)

    utils.plot_sweep(np.arange(0.0, FLAGS.p, 0.1), histories, 'p', 'loss')
    utils.plot_sweep(np.arange(0.0, FLAGS.p, 0.1), histories, 'p', 'acc')
    utils.plot_history(histories[0], 'cca')
    utils.plot_history(histories[-1], 'cca')
#   utils.plot_datapoints(dataset.x, torch.sign(m(dataset.x).squeeze()))

if __name__ == "__main__":
    app.run(main)
