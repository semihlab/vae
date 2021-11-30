import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Module

class ELBO(Module):
    def __init__(self, input_dim):
        super(ELBO, self).__init__()
        self.input_dim = input_dim
        self.dim = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]

    def forward(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(-1, self.dim), x.view(-1, self.dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

class IW_ELBO(Module):
    """
        Loss function of the importance weighted auto encoder
        WIP: Not yet debugged
    """
    def __init__(self, input_dim, num_samples):
        super(IW_ELBO, self).__init__()
        self.input_dim = input_dim
        self.num_samples = num_samples
        self.dim = self.input_dim[0] * self.input_dim[1] * self.input_dim[2]
        
    def forward(self, recon_x, x, mu, logvar):
        B, S, C, H, W = recon_x.size()
        x = x.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4) #[B x S x C x H x W]
        BCE = F.binary_cross_entropy(recon_x.view(B, S, self.dim), x.view(B, S, self.dim), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=2) # [B x S]
        # Get importance weight
        log_weight = BCE + KLD
        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim = -1)
        
        loss = torch.sum(torch.sum(weight * log_weight, dim=-1), dim=0)
        return loss
    
        