import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda
import numpy as np
import os

from model import construct_net_10d

def generate_artificial_data_10d(n_clusters, n_data_points):
    latent_means = torch.rand(n_clusters, 2)*10 - 5         # in range (-5, 5)
    latent_stds  = torch.rand(n_clusters, 2)*2.5 + 0.5      # in range (0.5, 3)
    
    latent_means /= 2
    latent_stds  /= 2
    
    labels = torch.randint(n_clusters, size=(n_data_points,))
    latent = latent_means[labels] + torch.randn(n_data_points, 2)*latent_stds[labels]
    latent = torch.cat([latent, torch.randn(n_data_points, 8)*1e-2], 1)
    
    random_transf = construct_net_10d('glow', init_identity=False)
    data = random_transf(latent).detach()
    
    return latent, data, labels


def make_dataloader(data, target, batch_size):
    dataset = BasicDataset(data, target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class BasicDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        super().__init__()
        self.data = data
        self.target = target
    
    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


def make_dataloader_emnist(batch_size, train=True, root_dir='./'):
    transform = Compose([ToTensor(),
                         Lambda(lambda x: x.view(1, 28, 28)),
                         Lambda(lambda x: x.transpose(1, 2))])
    try:
        emnist = torchvision.datasets.EMNIST(root=root_dir, split='digits', train=train, download=False, transform=transform)
    except RuntimeError:
        path = os.path.join(os.path.abspath(root_dir), 'EMNIST')
        yn = input(f'Dataset not found in {path}. Would you like to download it here? (y/n): ')
        while True:
            if yn not in ['y', 'n']:
                yn = input('Please type \'y\' or \'n\': ')
            else:
                if yn == 'y':
                    emnist = torchvision.datasets.EMNIST(root=root_dir, split='digits', train=train, 
                                    download=True, transform=transform)
                    break
                else:
                    print('Data will not be downloaded. Exiting script...')
                    quit()
    dataloader = torch.utils.data.DataLoader(emnist, batch_size=batch_size, shuffle=True)
    return dataloader


def get_mu_sig_emnist(model, dataloader):
    examples = iter(dataloader)
    data, target = next(examples)
    model.eval()
    latent = model(data.to(model.device)).detach().cpu()
    mu = torch.stack([latent[target == i].mean(0) for i in range(10)])
    sig = torch.stack([latent[target == i].std(0) for i in range(10)])
    return mu, sig 
    
    

