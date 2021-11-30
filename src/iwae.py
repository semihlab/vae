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
#         self.height, self.width, self.ch = input_dim
#         self.channels = channels
#         self.num_z = num_z
        
#         self.padding = 1
#         self.kernel_size = 3
#         self.stride = 2
#         self.dilation = 1
        
#         encoders = []
#         decoders = []
        
#         # output size: (width - kernel + padding * 2) / stride + 1
#         for i in range(len(self.channels)-1):
#             encoders.append(nn.Conv2d(in_channels=self.channels[i],
#                                       out_channels=self.channels[i + 1],
#                                       kernel_size=(self.kernel_size, self.kernel_size),
#                                       dilation=self.dilation,
#                                       padding=self.padding, stride=self.stride))
#             encoders.append(nn.GELU())
#             encoders.append(nn.Dropout(p=0.1))
            
#         # output size: (width - 1) * stride - padding * 2 + kernel + output_padding
#         for i in range(len(self.channels)-1, 0, -1):
#             decoders.append(nn.ConvTranspose2d(in_channels=self.channels[i],
#                                                out_channels=self.channels[i - 1],
#                                                kernel_size=(self.kernel_size+1, self.kernel_size+1),
#                                                dilation=self.dilation,
#                                                padding=self.padding, stride=self.stride))
#             decoders.append(nn.GELU())
#             decoders.append(nn.Dropout(p=0.1))

#         self.encoder = nn.Sequential(
#             *encoders,
#             nn.Flatten()
#         )
        
#         # compute encoder hidden size
#         enc_hidden_height = self.height
#         enc_hidden_width = self.width
#         for i in range(len(self.channels)-1):
#             enc_hidden_height = int((enc_hidden_height + 2*self.padding - self.dilation * (self.kernel_size-1) - 1)/self.stride + 1)
#             enc_hidden_width  = int((enc_hidden_width + 2*self.padding - self.dilation * (self.kernel_size-1) - 1)/self.stride + 1)
#         hidden_size = self.channels[-1] * enc_hidden_height * enc_hidden_width
        
#         # compute decoder hidden size
#         dec_hidden_height = enc_hidden_height
#         dec_hidden_width = enc_hidden_width
#         for i in range(len(self.channels)-1):
#             dec_hidden_height = int((dec_hidden_height-1) * self.stride - 2*self.padding + self.dilation*(self.kernel_size) + 1)
#             dec_hidden_width  = int((dec_hidden_width-1) * self.stride - 2*self.padding + self.dilation*(self.kernel_size) + 1)
        
#         self.z_mu = nn.Linear(hidden_size, self.num_z)
#         self.z_logvar = nn.Linear(hidden_size, self.num_z)
        
#         self.decoder = nn.Sequential(
#             nn.Linear(self.num_z, hidden_size),
#             nn.Unflatten(1, (self.channels[-1], enc_hidden_height, enc_hidden_width)),
#             *decoders,
#             nn.Flatten(),
#             nn.Linear(self.channels[0]*dec_hidden_height*dec_hidden_width, self.channels[0]*self.height*self.width),
#             nn.Unflatten(1, (self.channels[0], self.height, self.width))
#         )
        
# #         print(f'hidden: {hidden_height} x {hidden_width} = {hidden_size}')

        
#     def encode(self, x):
#         h = self.encoder(x)
#         return self.z_mu(h), self.z_logvar(h)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
        
#         return mu + eps*std

    def decode(self, z):
        B, _, _ = z.size()
        z = z.reshape(-1, self.num_z)  #[BS x D]
        x_bar = self.decoder(z)     #[BS x output_dim]
        x_bar = x_bar.view([B, -1, x_bar.size(1), x_bar.size(2), x_bar.size(3)]) #[B x S x C x H x W]
        return torch.sigmoid(x_bar)

    def forward(self, x, is_train=True):
        mu, logvar = self.encode(x)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x D]
        logvar = logvar.repeat(self.num_samples, 1, 1).permute(1, 0, 2) #[B x S x D]
        z = self.reparameterize(mu, logvar) #[B x S x D]
        recon_batch = self.decode(z)
        if not is_train:
            recon_batch = recon_batch[:, 0, :]
        return recon_batch, mu, logvar
