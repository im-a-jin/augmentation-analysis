import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistributionDropout:
    """
    Applies dropout to a given tensor given a distribution generator,
    mathematical transform, and application probability
    
    Args:
        dist (numpy.random.Generator.*): distribution to draw random numbers
        from
        p (float): probability of applying dropout given sample values
        m (lambda expr): mathematical transform to perform on sampled values,
        default identity
        kwargs: parameters for dist
    """
    def __init__(self, dist, p, m=None, **kwargs):
        self.dist = dist
        self.p = p
        if m is None:
            self.m = lambda x: x
        self.kwargs = kwargs

    def set_args(**kwargs):
        self.kwargs = kwargs

    def __call__(self, x):
        s = self.dist(**(self.kwargs), size=tuple(x.shape))
        ms = self.m(s)
        mask = ms < (1 - self.p)
        return x * mask
