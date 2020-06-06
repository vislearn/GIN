import torch
import torch.nn as nn
import numpy as np
from time import time
import os
from collections import OrderedDict

from data import make_dataloader, make_dataloader_emnist
from plot import artificial_data_reconstruction_plot, emnist_plot_samples, emnist_plot_spectrum, emnist_plot_variation_along_dims

import FrEIA.framework as Ff
import FrEIA.modules as Fm

class GIN(nn.Module):
    def __init__(self, dataset, n_epochs, epochs_per_line, lr, lr_schedule, batch_size, save_frequency, incompressible_flow, empirical_vars, data_root_dir='./', n_classes=None, n_data_points=None, init_identity=True):
        super().__init__()
        
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.epochs_per_line = epochs_per_line
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.save_frequency = min(save_frequency, n_epochs)
        self.incompressible_flow = incompressible_flow
        self.empirical_vars = empirical_vars
        self.init_identity = init_identity
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.timestamp = str(int(time()))
        
        if self.dataset == '10d':
            self.net = construct_net_10d(coupling_block='gin' if self.incompressible_flow else 'glow', init_identity=init_identity)
            assert type(n_classes) is int
            self.n_classes = n_classes
            self.n_dims = 10
            self.save_dir = os.path.join('./artificial_data_save/', self.timestamp)
            self.latent, self.data, self.target = generate_artificial_data_10d(self.n_classes, n_data_points)
            self.train_loader = make_dataloader(self.data, self.target, self.batch_size)
        elif self.dataset == 'EMNIST':
            if not init_identity:
                raise RuntimeError('init_identity=False not implemented for EMNIST experiments')
            self.net = construct_net_emnist(coupling_block='gin' if self.incompressible_flow else 'glow')
            self.n_classes = 10
            self.n_dims = 28*28
            self.save_dir = os.path.join('./emnist_save/', self.timestamp)
            self.data_root_dir = data_root_dir
            self.train_loader = make_dataloader_emnist(batch_size=self.batch_size, train=True, root_dir=self.data_root_dir)
            self.test_loader  = make_dataloader_emnist(batch_size=1000, train=False, root_dir=self.data_root_dir)
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
        
        if not empirical_vars:
            self.mu = nn.Parameter(torch.zeros(self.n_classes, self.n_dims).to(self.device)).requires_grad_()
            self.log_sig = nn.Parameter(torch.zeros(self.n_classes, self.n_dims).to(self.device)).requires_grad_()
        
        self.to(self.device)
            
    def forward(self, x, rev=False):
        x = self.net(x, rev=rev)
        return x
    
    def train_model(self):
        print(f'\nTraining model for {self.n_epochs} epochs \n')
        self.train()
        self.to(self.device)
        t0 = time()
        os.makedirs(os.path.join(self.save_dir, 'model_save'))
        os.makedirs(os.path.join(self.save_dir, 'figures'))
        print('  time     epoch    iteration         loss       last checkpoint')
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_schedule)
        losses = []
        for epoch in range(self.n_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
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
                self.print_loss(loss.item(), batch_idx, epoch, t0)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
            if (epoch+1)%self.epochs_per_line == 0:
                avg_loss = np.mean(losses)
                self.print_loss(avg_loss, batch_idx, epoch, t0, new_line=True)
                losses = []
            sched.step()
            if (epoch+1)%self.save_frequency == 0:
                self.save(os.path.join(self.save_dir, 'model_save', f'{epoch+1:03d}.pt'))
                self.make_plots()
    
    def print_loss(self, loss, batch_idx, epoch, t0, new_line=False):
        n_batches = len(self.train_loader)
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{self.n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
        else:
            last_save = (epoch//self.save_frequency)*self.save_frequency
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')
    
    def save(self, fname):
        state_dict = OrderedDict((k,v) for k,v in self.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)
    
    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['model'])
    
    def make_plots(self):
        if self.dataset == '10d':
            artificial_data_reconstruction_plot(self, self.latent, self.data, self.target)
        elif self.dataset == 'EMNIST':
            self.set_mu_sig()
            sig_rms = np.sqrt(np.mean((self.sig**2).detach().cpu().numpy(), axis=0))
            emnist_plot_samples(self, n_rows=20)
            emnist_plot_spectrum(self, sig_rms)
            n_dims_to_plot = 40
            top_sig_dims = np.flip(np.argsort(sig_rms))
            dims_to_plot = top_sig_dims[:n_dims_to_plot]
            emnist_plot_variation_along_dims(self, dims_to_plot)
        else:
            raise RuntimeError("Check dataset name. Doesn't match.")
    
    def set_mu_sig(self):
        if self.empirical_vars:
            examples = iter(self.test_loader)
            data, target = next(examples)
            self.eval()
            latent = self(data.to(self.device)).detach().cpu()
            self.mu = torch.stack([latent[target == i].mean(0) for i in range(10)])
            self.sig = torch.stack([latent[target == i].std(0) for i in range(10)])
        else:
            self.sig = self.log_sig.exp().detach()



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
        block = Fm.GINCouplingBlock
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
        block = Fm.GINCouplingBlock
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


# function is here rather than in data.py to prevent circular import
def generate_artificial_data_10d(n_clusters, n_data_points):
    latent_means = torch.rand(n_clusters, 2)*10 - 5         # in range (-5, 5)
    latent_stds  = torch.rand(n_clusters, 2)*2.5 + 0.5      # in range (0.5, 3)
    
    labels = torch.randint(n_clusters, size=(n_data_points,))
    latent = latent_means[labels] + torch.randn(n_data_points, 2)*latent_stds[labels]
    latent = torch.cat([latent, torch.randn(n_data_points, 8)*1e-2], 1)
    
    random_transf = construct_net_10d('glow', init_identity=False)
    data = random_transf(latent).detach()
    
    return latent, data, labels
