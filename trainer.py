import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, dataset, transforms, epochs, lr, device, test_all=False):
    history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            }
    n = len(dataset)
    split = int(n * 0.8)
    test_x, test_y = dataset[split:]
    test_mask = dataset.noise_mask[split:]
    if test_all:
        test_mask.fill_(1)
    test_x, test_y = test_x.to(device), test_y.to(device)
    test_x = test_x[np.argwhere(test_mask==1)]
    test_y = test_y[np.argwhere(test_mask==1)]
    for epoch in tqdm(range(1, epochs+1)):
        x, y = dataset[:split]
        x, y = x.to(device), y.to(device)
        y_ = model(transforms(x)).squeeze()
        
        loss = torch.mean(torch.log(1 + torch.exp(-y*y_)))
        pred = torch.sign(y_)
        acc = (y == pred).sum() / y.size(0)

        with torch.no_grad():
            test_y_ = model(test_x).squeeze()
            test_pred = torch.sign(test_y_)
            test_acc = (test_y == test_pred).float().mean()
            test_loss = torch.mean(torch.log(1 + torch.exp(-test_y*test_y_)))

        loss.backward()
        with torch.no_grad():
            model.linear1.weight -= model.linear1.weight.grad * lr
            model.linear1.bias -= model.linear1.bias.grad * lr
            model.linear1.weight.grad.zero_()
            model.linear1.bias.grad.zero_()

        history['train_loss'].append(loss.item())
        history['train_acc'].append(acc.item())
        history['test_loss'].append(test_loss.item())
        history['test_acc'].append(test_acc.item())

    for k in history.keys():
        history[k] = torch.Tensor(history[k])

    return model, history
