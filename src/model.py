import torch
from torch import nn
from torch.nn import Module

class VAE(Module):
    def __init__(self, layers, num_z):
        super(VAE, self).__init__()
        self.layers = layers
        self.num_z = num_z
        encoders = []
        decoders = []
        
        for i in range(len(self.layers) - 1):
            encoders.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            encoders.append(nn.ReLU())
            
        for i in range(len(self.layers) - 1, 0, - 1):
            decoders.append(nn.ReLU())
            decoders.append(nn.Linear(self.layers[i], self.layers[i - 1]))
        
        self.encoder = nn.Sequential(*encoders)
        
        self.z_mu = nn.Linear(self.layers[-1], self.num_z)
        self.z_logvar = nn.Linear(self.layers[-1], self.num_z)
        
        self.decoder = nn.Sequential(
            nn.Linear(self.num_z, self.layers[-1]),
            *decoders
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.z_mu(h), self.z_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std

    def decode(self, z):
        x_bar = self.decoder(z)
        return torch.sigmoid(x_bar)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.layers[0]))
        z = self.reparameterize(mu, logvar)
        recon_batch = self.decode(z)
        recon_batch = recon_batch.view(*x.shape)
        return recon_batch, mu, logvar
        
