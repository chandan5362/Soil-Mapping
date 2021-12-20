
import torch
import torch.nn as nn
class baseNPBlock(nn.Module):
    """relu non-linearities for NP block"""
    def __init__(self, inp_size,op_size, isnorm = True, bias = False, p = 0):
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
        bias : bool
                if True, bias is included for linear layer else discarded
        p : float
                probality to be considered while applying Dropout regularization
                
        """
        super().__init__()
        self.linear = nn.Linear(inp_size,op_size,bias = bias)
        self.relu  = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(inp_size)
        self.dropout = nn.Dropout2d(p)
        self.isnorm = isnorm
        self.tanh = nn.Tanh()
        
    def forward(self,x):
  
        if self.isnorm:
            x = self.batch_norm(x.permute(0,2,1)[:,:,:,None]) 
            x = self.dropout(x)
            x = x[:,:,:,0].permute(0,2,1)
        x = self.linear(x)
        x = self.relu(x)
        return x