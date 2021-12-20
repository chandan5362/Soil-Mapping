import torch
import torch.nn as nn
from attention import AttentionModule
from batch_mlp import batch_MLP


class LatentEncoder(nn.Module):
    def __init__(self,
                in_dim,
                 x_dim,
                hidden_dim = 32,
                latent_dim = 32,
                self_attn_type = 'dot',
                encoder_layer = 3,
                 n_multiheads = 8,
                min_std = 0.01,
                isnorm = True,
                p_encoder = 0,
                p_attn = 0,
                use_self_attn = False,
                attn_layers = 2,
                 rep ='mlp'
                ):
        
        super().__init__()
        
        self._use_attn = use_self_attn
        
        self.encoder = batch_MLP(in_dim, hidden_dim,hidden_dim, encoder_layer,isnorm, p_encoder)
        
        if self._use_attn:
            self.self_attn = AttentionModule(hidden_dim, self_attn_type, attn_layers,x_dim, rep = rep,isnorm = isnorm, p = p_attn, n_multiheads = n_multiheads)
        
        self.secondlast_layer = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.l_sigma = nn.Linear(hidden_dim, latent_dim) 
        self.min_std = min_std
        self.use_attn = use_self_attn
        
        self.relu = torch.nn.ReLU()
        
    def forward(self,x,y):
        encoder_inp = torch.cat([x,y], dim = -1) 
        encoded_op = self.encoder(encoder_inp)#(B, n, hd)

        if self.use_attn:
            encoded_op = self.self_attn(encoded_op, encoded_op, encoded_op) #(B, n, hd)
            

        mean_val = torch.mean(encoded_op, dim = 1) #mean aggregation over all the points (B, hd)
    
        #further MLP layer that maps parameters to gaussian latent
        mean_repr = self.relu(self.secondlast_layer(mean_val)) #(B, hd)

        μ = self.mean(mean_repr) # (B, ld)

        log_scale = self.l_sigma(mean_repr) #(B, ld)
        
        #to avoid mode collapse
        σ = self.min_std + (1-self.min_std)*torch.sigmoid(log_scale*0.5) #(B, ld)

        dist = torch.distributions.Normal(μ, σ)
        
        return dist
        