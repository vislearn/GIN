import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os

def artificial_data_reconstruction_plot(model, latent, data, target):
    """
    This function plots 8 figures of a reconstructed latent space, each for a different orientation of the 
    reconstructed latent space.
    """
    model.eval()
    model.cpu()
    z_reconstructed = model(data).detach()
    sig = torch.stack([z_reconstructed[target==i].std(0, unbiased=False) for i in range(model.n_classes)])
    rms_sig = np.sqrt(np.mean(sig.numpy()**2, 0))
    latent_sig = torch.stack([latent[target==i].std(0, unbiased=False) for i in range(model.n_classes)])
    latent_rms_sig = np.sqrt(np.mean(latent_sig.numpy()**2, 0))
    
    for dim_order in range(2):
        for dim1_factor in [1,-1]:
            for dim2_factor in [1,-1]:
                fig = plt.figure(figsize=(12, 3.5))
                
                plt.subplot(1, 4, 1)
                plt.scatter(latent[:,0], latent[:,1], c=target, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('GROUND TRUTH', fontsize=16, family='serif')
                
                plt.subplot(1, 4, 2)
                plt.scatter(data[:,0], data[:,1], c=target, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('OBSERVED DATA\n(PROJECTION)', fontsize=16, family='serif')
                
                plt.subplot(1, 4, 3)
                dim1 = np.flip(np.argsort(rms_sig))[dim_order]
                dim2 = np.flip(np.argsort(rms_sig))[(1+dim_order)%2]
                plt.scatter(dim1_factor*z_reconstructed[:,dim1], dim2_factor*z_reconstructed[:,dim2], c=target, s=6, alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('RECONSTRUCTION', fontsize=16, family='serif')
                
                plt.subplot(1, 4, 4)
                plt.semilogy(np.flip(np.sort(rms_sig)), '-ok')
                ground_truth = np.flip(np.sort(latent_rms_sig))
                plt.semilogy(scale_ground_truth(ground_truth, rms_sig), '-ok', alpha=0.3)
                plt.xticks([])
                plt.yticks([])
                plt.title('SPECTRUM', fontsize=16, family='serif')
                
                plt.tight_layout()
                fig_idx = 4*dim_order + 2*max(dim1_factor, 0) + max(dim2_factor, 0)
                plt.savefig(os.path.join(model.save_dir, 'figures', f'reconstruction_{fig_idx:d}.png'))
                plt.close()
    

def scale_ground_truth(y, x):
    logy = (np.log(y)-np.min(np.log(y))) * (np.max(np.log(x))-np.min(np.log(x))) 
    logy /= np.max(np.log(y))-np.min(np.log(y))
    logy += np.min(np.log(x))
    return np.exp(logy)



def emnist_plot_samples(model, n_rows, dims_to_sample=torch.arange(784), temp=1):
    """
    Plots sampled digits. Each row contains all 10 digits with a consistent style
    """
    model.eval()
    fig = plt.figure(figsize=(10, n_rows))
    n_dims_to_sample = len(dims_to_sample)
    style_sample = torch.zeros(n_rows, 784)
    style_sample[:,dims_to_sample] = torch.randn(n_rows, n_dims_to_sample)*temp
    style_sample = style_sample.to(model.device)
    # style sample: (n_rows, n_dims)
    # mu,sig: (n_classes, n_dims)
    # latent: (n_rows, n_classes, n_dims)
    latent = style_sample.unsqueeze(1)*model.sig.unsqueeze(0) + model.mu.unsqueeze(0)
    latent.detach_()
    # data: (n_rows, n_classes, 28, 28)
    data = (model(latent.view(-1, 784), rev=True)[0]).detach().cpu().numpy().reshape(n_rows, 10, 28, 28)
    im = data.transpose(0, 2, 1, 3).reshape(n_rows*28, 10*28)
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'samples.png'), bbox_inches='tight', pad_inches=0.5)
    plt.close()


def emnist_plot_variation_along_dims(model, dims_to_plot):
    """
    Makes a plot for each of the given latent space dimensions. Each column contains all 10 digits
    with a consistent style. Each row shows the effect of varying the latent space value of the 
    chosen dimension from -2 to +2 standard deviations while keeping the latent space
    values of all other dimensions constant at the mean value. The rightmost column shows a heatmap
    of the absolute pixel difference between the column corresponding to -1 std and +1 std
    """
    os.makedirs(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots'))
    max_std = 2
    n_cols = 9
    model.eval()
    for i, dim in enumerate(dims_to_plot):
        fig = plt.figure(figsize=(n_cols+1, 10))
        style = torch.zeros(n_cols, 784)
        style[:, dim] = torch.linspace(-max_std, max_std, n_cols)
        style = style.to(model.device)
        # style: (n_cols, n_dims)
        # mu,sig: (n_classes, n_dims)
        # latent: (n_classes, n_cols, n_dims)
        latent = style.unsqueeze(0)*model.sig.unsqueeze(1) + model.mu.unsqueeze(1)
        latent.detach_()
        data = (model(latent.view(-1, 784), rev=True)[0]).detach().cpu().numpy().reshape(10, n_cols, 28, 28)
        im = data.transpose(0, 2, 1, 3).reshape(10*28, n_cols*28)
        # images at +1 and -1 std
        im_p1 = im[:, 28*2:28*3]
        im_m1 = im[:, 28*6:28*7]
        # full image with spacing between the two parts
        im = np.concatenate([im, np.ones((10*28, 3)), np.abs(im_p1-im_m1)], axis=1)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'variation_plots', f'variable_{i+1:03d}.png'), 
                    bbox_inches='tight', pad_inches=0.5)
        plt.close()


def emnist_plot_spectrum(model, sig_rms):
    fig = plt.figure(figsize=(12, 6))
    plt.semilogy(np.flip(np.sort(sig_rms)), 'k')
    plt.xlabel('Latent dimension (sorted)')
    plt.ylabel('Standard deviation (RMS across classes)')
    plt.title('Spectrum on EMNIST')
    plt.savefig(os.path.join(model.save_dir, 'figures', f'epoch_{model.epoch+1:03d}', 'spectrum.png'))
    plt.close()
    



