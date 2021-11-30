import torch
from torch import nn
from torch.nn import Module

from src.vae import VAE

class IWAE(VAE):
    def __init__(self, input_dim, channels, num_z, num_samples):
        super(IWAE, self).__init__(input_dim=input_dim,
                                   channels=channels,
                                   num_z=num_z)
        self.num_samples = num_samples

    def decode(self, z):
        B, _, _ = z.size()
        z = z.contiguous()
        z = z.view(-1, self.num_z)  #[BS x D]
        x_bar = self.decoder(z)     #[BS x output_dim]
        x_bar = x_bar.view([B, -1, x_bar.size(1), x_bar.size(2), x_bar.size(3)]) #[B x S x C x H x W]
        return torch.sigmoid(x_bar)

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x D]
        logvar = logvar.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x D]
        z = self.reparameterize(mu, logvar) #[B x S x D]
        recon_batch = self.decode(z)
        return recon_batch, mu, logvar
    
    @torch.no_grad()
    def reconstruct(self, x):
        mu, logvar = self.encode(x)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x D]
        logvar = logvar.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x D]
        z = self.reparameterize(mu, logvar) #[B x S x D]
        recon_batch = self.decode(z)
        return recon_batch[:, 0, :]
