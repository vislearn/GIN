import argparse
import os
from model import GIN

parser = argparse.ArgumentParser(description='Experiments on EMNIST with GIN (training script)')
parser.add_argument('--n_epochs', type=int, default=100,
                    help='Number of training epochs (default 100)')
parser.add_argument('--epochs_per_line', type=int, default=1,
                    help='Print a new line after this many epochs (default 1)')
parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learn rate (default 3e-4)')
parser.add_argument('--lr_schedule', nargs='+', type=int, default=[50],
                    help='Learn rate schedule (decrease lr by factor of 10 at these epochs, default [50]). \
                            Usage example: --lr_schedule 20 40')
parser.add_argument('--batch_size', type=int, default=240,
                    help='Batch size (default 240)')
parser.add_argument('--save_frequency', type=int, default=10,
                    help='Save a new checkpoint and make plots after this many epochs (default 10)')
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

model = GIN(dataset='EMNIST', 
            n_epochs=args.n_epochs, 
            epochs_per_line=args.epochs_per_line, 
            lr=args.lr, 
            lr_schedule=args.lr_schedule, 
            batch_size=args.batch_size, 
            save_frequency=args.save_frequency, 
            data_root_dir=args.data_root_dir, 
            incompressible_flow=args.incompressible_flow, 
            empirical_vars=args.empirical_vars)
model.train_model()




