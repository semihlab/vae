import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module

class ELBO(Module):
    def __init__(self, input_dim):
        super(ELBO, self).__init__()
        self.input_dim = input_dim

    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, self.input_dim), x.view(-1, self.input_dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

