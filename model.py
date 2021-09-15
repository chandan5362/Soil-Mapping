import torch
import torch.nn as nn
import torch.nn.functional as F

#local modules
from . import  LatentEncoder, DeterministicEncoder, Decoder



class LatentModel(nn.Module):
    def __init__(self,
               x_dim,
               y_dim,
               hidden_dim = 32,
               latent_dim = 32,
               latent_self_attn_type = 'dot',
               det_self_attn_type = 'dot',
               det_cross_attn_type = 'multihead',
               n_multiheads = 8,
               n_lat_enc_layer = 2,
               n_det_enc_layer = 2,
               n_decoder_layer = 2,
               rep ='mlp',
               use_deterministic_enc = False,
               min_std = 0.01,
               p_drop = 0,
               isnorm = True,
               p_attn_drop = 0,
               attn_layers = 2,
               use_self_attn = False,
               context_in_target = True
                ):
        
        super().__init__()
        self.laten_encoder = LatentEncoder(x_dim+y_dim,
                                           x_dim,
                                           hidden_dim=hidden_dim,
                                           latent_dim=latent_dim,
                                           self_attn_type=latent_self_attn_type,
                                           encoder_layer=n_lat_enc_layer,
                                           n_multiheads = n_multiheads,
                                           min_std=min_std,
                                           isnorm = isnorm,
                                           p_encoder=p_drop,
                                           p_attn=p_attn_drop,
                                           rep = 'identity',
                                           use_self_attn=use_self_attn,
                                           attn_layers=attn_layers 
                                          )
        self.deterministic_encoder = DeterministicEncoder(x_dim+y_dim,
                                                          x_dim,
                                                          isnorm = isnorm,
                                                          hidden_dim=hidden_dim,
                                                          n_encoder_layer=n_det_enc_layer,
                                                          rep=rep,
                                                          self_attn_type=det_self_attn_type,
                                                          cross_attn_type=det_cross_attn_type,
                                                          p_encoder=p_drop,
                                                          p_attention=p_attn_drop,
                                                          attn_layers=attn_layers,
                                                          use_self_attn=use_self_attn,
                                                          n_multiheads = n_multiheads
                                                         )
        self.decoder = Decoder(x_dim,
                              y_dim,
                              hidden_dim  = hidden_dim,
                              latent_dim=latent_dim,
                              n_decoder_layer=n_decoder_layer,
                              use_deterministic_path=use_deterministic_enc,
                              min_std=min_std,
                              isnorm=isnorm,
                              dropout_p=p_drop
                              )
        self.use_deterministic_enc = use_deterministic_enc
        self.context_in_target = context_in_target
        
        
    def forward(self, c_x, c_y, t_x, t_y = None, training = False):
        dist_prior = self.laten_encoder(c_x, c_y)

        if t_y is not None:
            dist_posterior = self.laten_encoder(t_x, t_y)
            z = dist_posterior.sample() #(B, ld)
        else:
            z = dist_prior.sample() #(B, ld)
            
        n_target = t_x.shape[1]
        z = z.unsqueeze(1).repeat(1, n_target,1) #(B, n_target, 1)
        
        
        
        if self.use_deterministic_enc:
            r = self.deterministic_encoder(c_x, c_y, t_x) #(B, n_target=m, H)
            representation = torch.cat([r, z], axis = -1) #(B, ld+hd)
        else:
            representation = z
#  
            
        dist, μ, σ = self.decoder(representation, t_x)

        #at test time, target y is not Known so we return None
        if t_y is not None:
            log_p = dist.log_prob(t_y) #(B, m, 1)
            
            kl_loss = torch.distributions.kl_divergence(dist_posterior, dist_prior).sum(dim = -1, keepdim = True)
            kl_loss = torch.tile(kl_loss, [1, n_target])[:,:, None]

            loss  = -(torch.mean(log_p - kl_loss/n_target))

            mse_loss = F.mse_loss(dist.loc, t_y, reduction = 'none')[:,:c_x.size(1)].mean()
        else:
            kl_loss = None
            log_p = None
            mse_loss = None
            loss = None
            
        y_pred =  dist.loc
            
        return y_pred,  dict(loss = loss, loss_p = log_p, loss_kl = kl_loss, mse_loss = mse_loss), dist



Regressor = LatentModel(2,1,
                    p_drop = 0.0,
                    p_attn_drop=0.5,
                    hidden_dim = 128,
                    latent_dim = 128,
                    n_decoder_layer = 4,
                    n_lat_enc_layer=4,
                    n_det_enc_layer=4,
                    n_multiheads = 1,
                    isnorm = True,
                    use_self_attn=True,
                    use_deterministic_enc=True,
                    context_in_target= False
                        )