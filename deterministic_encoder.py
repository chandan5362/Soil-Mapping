

import torch
import torch.nn as nn
from batch_mlp import batch_MLP
from attention import AttentionModule


class DeterministicEncoder(nn.Module):
    def __init__(
                self,
                in_dim,
                x_dim,
                isnorm = True,
                hidden_dim = 32,
                n_encoder_layer = 2,
                rep = 'mlp',
                self_attn_type ='dot',
                cross_attn_type ='dot',
                n_multiheads = 8,
                p_encoder = 0,
                p_attention = 0,
                attn_layers = 2,
                use_self_attn = False
                ):
        super().__init__()
        
        self.use_self_attn = use_self_attn
        
        self.encoder = batch_MLP(in_dim, hidden_dim, hidden_dim, n_encoder_layer,isnorm, p_encoder)
        
        if self.use_self_attn:
            self.self_attn = AttentionModule(hidden_dim, self_attn_type, attn_layers,x_dim, rep = 'identitiy',isnorm = isnorm, p = p_attention, n_multiheads = n_multiheads)
            
        self.cross_attn = AttentionModule(hidden_dim, cross_attn_type, attn_layers, x_dim, rep ='mlp', isnorm = isnorm, p = p_attention, n_multiheads = n_multiheads)
        
    
    def forward(self, context_x, context_y, target_x):
        #concatenate context_x, context_y along the last dim.
        det_enc_in = torch.cat([context_x, context_y], dim = -1)
        
        det_encoded = self.encoder(det_enc_in) #(B, n, hd)
        if self.use_self_attn:
            det_encoded = self.self_attn(det_encoded, det_encoded, det_encoded) #(B, n, hd)

        h = self.cross_attn(context_x, target_x, det_encoded) #(B, n, hd)
        
        return h
        
        