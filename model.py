import torch
import torch.nn as nn
import numpy as np
from time import time
import os
from collections import OrderedDict

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from gin_coupling_block import GINCouplingBlock

class GIN(nn.Module):
    def __init__(self, dataset, incompressible_flow=True, n_classes=None, empirical_vars=True, init_identity=True):
        super().__init__()
        
        self.dataset = dataset
        self.incompressible_flow = incompressible_flow
        self.empirical_vars = empirical_vars
        self.init_identity = init_identity
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.dataset == '10d':
            self.net = construct_net_10d(coupling_block='gin' if self.incompressible_flow else 'glow', init_identity=init_identity)
            assert type(n_classes) is int
            self.n_classes = n_classes
            self.n_dims = 10
        elif self.dataset == 'EMNIST':
            if not init_identity:
                raise RuntimeError('init_identity=False not implemented for EMNIST experiments')
            self.net = construct_net_emnist(coupling_block='gin' if self.incompressible_flow else 'glow')
            self.n_classes = 10
            self.n_dims = 28*28
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
        
        if not empirical_vars:
            self.mu = nn.Parameter(torch.zeros(self.n_classes, self.n_dims).to(self.device)).requires_grad_()
            self.log_sig = nn.Parameter(torch.zeros(self.n_classes, self.n_dims).to(self.device)).requires_grad_()
        
        self.to(self.device)
            
    def forward(self, x, rev=False):
        x = self.net(x, rev=rev)
        return x
    
    def train_model(self, dataloader, n_epochs, lr, lr_schedule=[], epochs_per_line=1, 
                    save_frequency=None, save_dir='./model_save/'):
        self.train()
        self.to(self.device)
        t0 = time()
        os.makedirs(os.path.join(save_dir, str(int(t0))))
        if save_frequency is None:
            save_frequency = n_epochs
        save_frequency = min(save_frequency, n_epochs)
        print('  time     epoch    iteration         loss       last checkpoint')
        optimizer = torch.optim.Adam(self.parameters(), lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_schedule)
        losses = []
        for epoch in range(n_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                if self.empirical_vars:
                    # first check that std will be well defined
                    if min([sum(target==i).item() for i in range(self.n_classes)]) < 2:
                        # don't calculate loss and update weights -- it will give nan or error
                        # go to next batch
                        continue
                optimizer.zero_grad()
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                z = self.net(data)          # latent space variable
                logdet_J = self.net.log_jacobian(run_forward=False)
                if self.empirical_vars:
                    # we only need to calculate the std
                    sig = torch.stack([z[target==i].std(0, unbiased=False) for i in range(self.n_classes)])
                    # negative log-likelihood for gaussian in latent space
                    loss = 0.5 + sig[target].log().mean(1) + 0.5*np.log(2*np.pi)
                else:
                    m = self.mu[target]
                    ls = self.log_sig[target]
                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(0.5*(z-m)**2 * torch.exp(-2*ls) + ls, 1) + 0.5*np.log(2*np.pi)
                loss -= logdet_J / self.n_dims
                loss = loss.mean()
                self.print_loss(loss.item(), batch_idx, len(dataloader), epoch, n_epochs, t0, save_frequency)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
            if (epoch+1)%epochs_per_line == 0:
                avg_loss = np.mean(losses)
                self.print_loss(avg_loss, batch_idx, len(dataloader), epoch, n_epochs, t0, save_frequency, new_line=True)
                losses = []
            sched.step()
            if (epoch+1)%save_frequency == 0:
                self.save(os.path.join(save_dir, str(int(t0)), f'{epoch+1:03d}.pt'))
    
    def print_loss(self, loss, batch_idx, n_batches, epoch, n_epochs, t0, save_frequency, new_line=False):
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
        else:
            last_save = (epoch//save_frequency)*save_frequency
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')
    
    def save(self, fname):
        state_dict = OrderedDict((k,v) for k,v in self.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)
    
    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['model'])



def subnet_fc_10d(c_in, c_out, init_identity):
    subnet = nn.Sequential(nn.Linear(c_in, 10), nn.ReLU(),
                            nn.Linear(10, 10), nn.ReLU(),
                            nn.Linear(10,  c_out))
    if init_identity:
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_net_10d(coupling_block, init_identity=True):
    if coupling_block == 'gin':
        block = GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock
    
    nodes = [Ff.InputNode(10, name='input')]
    
    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':lambda c_in,c_out: subnet_fc_10d(c_in, c_out, init_identity), 'clamp':2.0},
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                        {'seed':np.random.randint(2**31)},
                        name=F'permute_{k+1}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)


def subnet_fc(c_in, c_out):
    width = 392
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv1(c_in, c_out):
    width = 16
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv2(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def construct_net_emnist(coupling_block):
    if coupling_block == 'gin':
        block = GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock
    
    nodes = [Ff.InputNode(1, 28, 28, name='input')]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv1, 'clamp':2.0},
                             name=F'coupling_conv1_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv1_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample2'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv2, 'clamp':2.0},
                             name=F'coupling_conv2_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv2_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    for k in range(2):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_fc, 'clamp':2.0},
                             name=F'coupling_fc_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_fc_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)

