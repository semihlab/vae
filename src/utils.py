import torch
from torch import optim, nn
from torch.nn import functional as F
import torchvision

import numpy as np

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

## Train related
def train(epoch,
          model, train_loader,
          criterion, optimizer, scheduler,
          device="cpu"):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = criterion(recon_batch, data, mu, logvar)
        train_loss += loss.item() / len(train_loader.dataset)
        loss.backward()
        optimizer.step()
    scheduler.step()
        
    return train_loss

@torch.no_grad()
def reconstruct(model, test_loader, device="cpu"):
    model.eval()
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch = model.reconstruct(data)

            n = min(data.shape[0], 8)
            samples = data[:n].cpu().numpy()
            recons = recon_batch[:n].cpu().numpy()
            
            break

    return samples, recons

@torch.no_grad()
def test(epoch,
         model, test_loader,
         criterion,
         device="cpu"):
    model.eval()
    test_elbo_loss = 0
    test_mse_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            
            elbo_loss = criterion(recon_batch, data, mu, logvar)
            test_elbo_loss += elbo_loss.item() / len(test_loader.dataset)
            
            # an exception for iwae
            if (len(recon_batch.size()) == 5):
                recon_batch = recon_batch[:, 0, :]
            mse_loss = F.mse_loss(recon_batch, data)
            test_mse_loss += mse_loss.item() / len(test_loader.dataset)
    
    return test_elbo_loss, test_mse_loss

## Viz related
def to_rgb(sample):
    r = sample[0]
    g = sample[1]
    b = sample[2]
    rgb = (np.dstack((r,g,b)) * 255.999) .astype(np.uint8)
    return rgb

def visualize_imgs(samples, recons):
    (n,c,h,w) = samples.shape
    plt.figure(figsize=(28, 8))
    
    if (c == 3):
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(to_rgb(samples[i]))

            plt.subplot(2, n, i + 1 + n)
            plt.imshow(to_rgb(recons[i]))
        plt.show()
    elif (c == 1):
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(samples[i].reshape(28, 28), cmap='gray_r')

            plt.subplot(2, n, i + 1 + n)
            plt.imshow(recons[i].reshape(28, 28), cmap='gray_r')
        plt.show()
    