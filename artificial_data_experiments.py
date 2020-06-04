import argparse
from model import GIN
from data import generate_artificial_data_10d, make_dataloader
from plot import artificial_data_reconstruction_plot

parser = argparse.ArgumentParser(description='Artificial data experiments (2d data embedded in 10d) with GIN.')
parser.add_argument('--n_clusters', type=int, default=5,
                    help='Number of components in gaussian mixture (default 5)')
parser.add_argument('--n_data_points', type=int, default=10000,
                    help='Number of data points in artificial data set (default 10000)')
parser.add_argument('--n_epochs', type=int, default=80,
                    help='Number of training epochs (default 80)')
parser.add_argument('--epochs_per_line', type=int, default=5,
                    help='Print a new line after this many epochs (default 5)')
parser.add_argument('--lr', type=float, default=1e-2,
                    help='Learn rate (default 1e-2)')
parser.add_argument('--lr_schedule', nargs='+', type=int, default=[50],
                    help='Learn rate schedule (decrease lr by factor of 10 at these epochs, default [50]). \
                            Usage example: --lr_schedule 20 40')
parser.add_argument('--batch_size', type=int, default=1000,
                    help='Batch size (default 1000)')
parser.add_argument('--incompressible_flow', type=int, default=1,
                    help='Use an incompressible flow (GIN) (1, default) or compressible flow (GLOW) (0)')
parser.add_argument('--empirical_vars', type=int, default=1,
                    help='Estimate empirical variables (means and stds) for each batch (1, default) or learn them along \
                            with model weights (0)')
parser.add_argument('--init_identity', type=int, default=1,
                    help='Initialize the network as the identity (1, default) or not (0)')
args = parser.parse_args()
assert args.incompressible_flow in [0,1], 'Argument should be 0 or 1'
assert args.empirical_vars in [0,1], 'Argument should be 0 or 1'
assert args.init_identity in [0,1], 'Argument should be 0 or 1'

print('Initializing GIN model \n')
model = GIN(dataset='10d', incompressible_flow=args.incompressible_flow, n_classes=args.n_clusters, 
            empirical_vars=args.empirical_vars, init_identity=args.init_identity)
n_params = sum(p.numel() for p in model.parameters())
print(f'Number of trainable parameters: {n_params} \n')
print('Creating artificial data \n')
latent, data, target = generate_artificial_data_10d(args.n_clusters, args.n_data_points)
dataloader = make_dataloader(data, target, args.batch_size)
print(f'Training model for {args.n_epochs} epochs \n')
model.train_model(dataloader, args.n_epochs, args.lr, lr_schedule=args.lr_schedule, 
                    epochs_per_line=args.epochs_per_line, save_dir='./model_save/artificial_data/')
print('\nDone training. Saving model and plotting reconstruction \n')
artificial_data_reconstruction_plot(model, latent, data, target)


