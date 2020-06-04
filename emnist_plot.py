import argparse
import os
import numpy as np
from time import time
from model import GIN
from data import make_dataloader_emnist, get_mu_sig_emnist
from plot import emnist_plot_samples

parser = argparse.ArgumentParser(description='Experiments on EMNIST with GIN (plotting script)')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to .pt checkpoint file to be loaded. Defaults to final save of most recent training run')
parser.add_argument('--data_root_dir', type=str, default='./',
                    help='Directory in which \'EMNIST\' directory storing data is located (defaults to current directory). If the data is not found here you will be prompted to download it')
parser.add_argument('--incompressible_flow', type=int, default=1,
                    help='Use an incompressible flow (GIN) (1, default) or compressible flow (GLOW) (0)')
parser.add_argument('--empirical_vars', type=int, default=1,
                    help='Estimate empirical variables (means and stds) for each batch (1, default) or learn them along \
                            with model weights (0)')
args = parser.parse_args()
assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'

model = GIN(dataset='EMNIST', incompressible_flow=args.incompressible_flow, 
            empirical_vars=args.empirical_vars)
if args.checkpoint_path is not None:
    fname = args.checkpoint_path
else:
    save_dir = './model_save/emnist/'
    t_array = np.array([int(t) for t in os.listdir(save_dir)])
    last_t = str(np.max(t_array))
    pt_array = np.array([int(pt.split('.')[0]) for pt in os.listdir(os.path.join(save_dir, last_t))])
    last_pt = f'{np.max(pt_array):03d}.pt'
    fname = os.path.join(save_dir, last_t, last_pt)
    assert os.path.isfile(fname), f'No file called {fname} exists'
model.load(fname)
dataloader = make_dataloader_emnist(batch_size=1000, train=False, root_dir=args.data_root_dir)
mu, sig = get_mu_sig_emnist(model, dataloader)
t = int(time())             # time stamp which will be used as save name
emnist_plot_samples(model, 12, t, mu=mu, sig=sig)
