import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, dataset, transforms, lr, epochs, device):
    history = {
            'train_loss': [], 'train_acc': [], 'train_cca': [],
            'test_loss': [], 'test_acc': [], 'test_cca': [],
            }
    split = int(len(dataset) * 0.8)
    test_x, test_y = dataset[split:]
    test_x, test_y = test_x.to(device), test_y.to(device)
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
            test_acc = (test_y == test_pred).sum() / test_y.size(0)
            test_loss = torch.mean(torch.log(1 + torch.exp(-test_y*test_y_)))

        loss.backward()
        with torch.no_grad():
            model.linear1.weight -= model.linear1.weight.grad * lr
            model.linear1.bias -= model.linear1.bias.grad * lr
            model.linear1.weight.grad.zero_()
            model.linear1.bias.grad.zero_()

        history['train_loss'].append(loss.item())
        history['train_acc'].append(acc.item())
        history['train_cca'].append(1-acc.item())
        history['test_loss'].append(test_loss.item())
        history['test_acc'].append(test_acc.item())
        history['test_cca'].append(1-test_acc.item())

#   print(torch.vstack((test_y, test_y_, test_pred)).T)

#   print('train loss =', history['train_loss'][-1])
#   print('test loss =', history['test_loss'][-1])
#   print('train acc =', history['train_acc'][-1])
#   print('test acc =', history['test_acc'][-1])
#   print('train cca =', history['train_cca'][-1])
#   print('test cca =', history['test_cca'][-1])

    return model, history
