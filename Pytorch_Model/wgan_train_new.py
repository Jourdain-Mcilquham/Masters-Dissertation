import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models import Generator, Discriminator, weights_init
from dataset import SpectralDataset
import torch.autograd as autograd
import torch.nn.functional as F

# Hyperparameters and configuration settings
beta1 = 0
beta2 = 0.9
p_coeff = 10
n_critic = 5
lr = 1e-5
input_dim = 50
epoch_num = 100
batch_size = 256
nz = 50  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Modified wgan training function from: https://github.com/LixiangHan/GANs-for-1D-Signal
'''

def main():
    # Create save directory if it doesn't exist
    save_dir = "different_noise_outputs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Loading spectral dataset
    ds = SpectralDataset(batch_size=batch_size, data_location="ratio_training2.h5")
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(25, input_dim, device=device)

    # Initialize discriminator and generator networks
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    netG = Generator().to(device)
    netG.apply(weights_init)

    # Load weights if they exist
    generator_weights_path = 'nets/wgan_gp_netG.pkl'
    discriminator_weights_path = 'nets/wgan_gp_netD.pkl'

    if os.path.exists(generator_weights_path):
        netG = torch.load(generator_weights_path)
        print("Loaded generator weights")

    if os.path.exists(discriminator_weights_path):
        netD = torch.load(discriminator_weights_path)
        print("Loaded discriminator weights")

    # Defining optimizers
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    # Training loop
    for epoch in range(epoch_num):
        for step, data in enumerate(dl):
            # Training discriminator
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            netD.zero_grad()

            noise = torch.randn(b_size, input_dim, device=device)
            fake = netG(noise)

            # Gradient penalty for WGAN
            eps = torch.Tensor(b_size, 1).uniform_(0, 1).to(device)
            x_p = eps * data.to(device) + (1 - eps) * fake
            grad = autograd.grad(netD(x_p).mean(), x_p, create_graph=True, retain_graph=True)[0].view(b_size, -1)
            grad_norm = torch.norm(grad, 2, 1)
            grad_penalty = p_coeff * torch.pow(grad_norm - 1, 2)

            loss_D = torch.mean(netD(fake) - netD(real_cpu))
            loss_D.backward()
            optimizerD.step()

            # Weight clipping for discriminator
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)

            if step % n_critic == 0:
                # Training generator
                noise = torch.randn(b_size, input_dim, device=device)
                netG.zero_grad()
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))
                loss_G.backward()
                optimizerG.step()

            # Logging training information
            if step % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch, epoch_num, step, len(dl), loss_D.item(), loss_G.item()))

        # Save training process images with fixed noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            f, a = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(4):
                for j in range(4):
                    a[i][j].plot(fake[i * 4 + j].view(-1))
                    a[i][j].set_xticks(())
                    a[i][j].set_yticks(())
            plt.savefig('img/wgan_gp_epoch_%d.png' % epoch)
            plt.close()
        
        # Save outputs using different noise vectors for visualization
        num_different_noises = 5
        for noise_idx in range(num_different_noises):
            noise = torch.randn(25, input_dim, device=device)
            with torch.no_grad():
                fake = netG(noise).detach().cpu()
                f, a = plt.subplots(4, 4, figsize=(8, 8))
                for i in range(4):
                    for j in range(4):
                        a[i][j].plot(fake[i * 4 + j].view(-1))
                        a[i][j].set_xticks(())
                        a[i][j].set_yticks(())
                plt.savefig(os.path.join(save_dir, f'wgan_gp_epoch_{epoch}_noise_{noise_idx}.png'))
                plt.close()

    # Save model weights
    torch.save(netG.state_dict(), 'nets/wgan_gp_netG.pkl')  # Save only the state_dict, not the entire model
    torch.save(netD.state_dict(), 'nets/wgan_gp_netD.pkl')  # Save only the state_dict, not the entire model

if __name__ == '__main__':
    main()
