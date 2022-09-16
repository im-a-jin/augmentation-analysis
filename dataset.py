import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset

class XORMixture(Dataset):
    def __init__(self, dims, var, n, noise_rate, mu=None):
        if not mu:
            mu1 = np.random.rand(dims)
        else:
            mu1 = np.array(mu)
        mu1 /= np.linalg.norm(mu1)
        mu2 = np.random.rand(dims)
        mu2 -= mu1.dot(mu2) * mu1 / np.linalg.norm(mu1)**2
        mu2 /= np.linalg.norm(mu2)

        assert(abs(mu1.dot(mu2) < 1e-5))

        self.mu = np.array([mu1, -mu1, mu2, -mu2])
        self.var = var * np.eye(mu1.size)
        self.n = n
        self.noise_rate = noise_rate
        
        self._generate_samples()

    def _generate_samples(self):
        x = []
        for i in range(4):
           x.append(np.random.multivariate_normal(
               self.mu[i], self.var, self.n // 4))
        self.x = torch.Tensor(np.vstack(x))
        self.y = torch.Tensor(np.hstack((np.ones(self.n // 2),
                                        -np.ones(self.n // 2))))
        self.noise_mask = 2 * (torch.rand(self.n) < 1 - self.noise_rate).int() - 1
        self.y = self.y * self.noise_mask

        self._shuffle()

    def _shuffle(self):
        rand_idx = torch.randperm(self.n)
        self.x = self.x[rand_idx, :]
        self.y = self.y[rand_idx]
        self.noise_mask = self.noise_mask[rand_idx]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]

    def __len__(self):
        return len(self.y)
