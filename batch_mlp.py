import torch
import torch.nn as nn

from base_mlp import baseNPBlock
class batch_MLP(nn.Module):
    """ Batch MLP layer for NP-Encoder"""
    def __init__(self, in_size, op_size, last_op_size,  num_layers, isnorm = True, p = 0):
        """init function for linear2d class
        
        parameters
        ----------
        inp_size : int
                input dimension for the Encoder part (d_in)
        op_size : int
                output dimension for Encoder part(d_out)
        norm : str
                normalization to be applied on linear output
                pass norm == 'batch' to apply batch normalization
                else dropout normalization is applied
                
        return torch.tensor of size (B,num_context_points,d_out)
        """
        super().__init__()
        self.in_size = in_size
        self.op_size = op_size
        self.num_layers = num_layers
        self.isnorm  = isnorm
        
        self.first_layer = baseNPBlock(in_size, op_size, self.isnorm, False,p)
        self.encoder = nn.Sequential(*[baseNPBlock(op_size, op_size, self.isnorm, False, p) for layer in range(self.num_layers-2)])
        self.last_layer = nn.Linear(op_size, last_op_size )
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.encoder(x)
        x = self.last_layer(x)
        
        return x