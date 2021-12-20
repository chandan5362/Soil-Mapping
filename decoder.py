import torch
import torch.nn as nn
import torch.nn.functional as F
from batch_mlp import batch_MLP


class Decoder(nn.Module):
    def __init__(self,
                 x_dim,
                 y_dim,
                 hidden_dim = 32,
                 latent_dim = 32,
                 n_decoder_layer = 3,
                 use_deterministic_path = True,
                 min_std = 0.01,
                 isnorm = True,
                 dropout_p = 0,
                ):
        super().__init__()
        
        self.isnorm = isnorm
        self.target_transform = nn.Linear(x_dim, hidden_dim)
        
        if use_deterministic_path:
            hidden_dim_2 = hidden_dim + latent_dim + x_dim
        else:
            hidden_dim_2 = latent_dim + x_dim
            
        self.decoder = batch_MLP(hidden_dim_2, hidden_dim_2, 2, n_decoder_layer, isnorm, dropout_p)
        

        self.deterministic_path = use_deterministic_path
        self.min_std = min_std
        
        
    def forward(self, rep,  t_x):

        hidden_decode = torch.cat([rep,t_x], dim = -1)

        hidden_decode = self.decoder(hidden_decode)
        mean, log_sigma = hidden_decode[:, : ,:1],hidden_decode[:, :, 1:] 
        
        #clamp sigma
        sigma = self.min_std + (1 - self.min_std) * F.softplus(log_sigma)
        
        dist = torch.distributions.Normal(mean,sigma)
        
        return dist, mean, sigma