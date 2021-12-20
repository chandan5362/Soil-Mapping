import torch
import numpy as np
from model import LatentModel
from sklearn.metrics import r2_score
import pandas as pd

Regressor = LatentModel(2,1,
                        p_drop = 0,
                        p_attn_drop=0.3,
                        hidden_dim = 128,
                        latent_dim = 128,
                        n_decoder_layer = 2,
                        n_lat_enc_layer=4,
                        n_det_enc_layer=4,
                        n_multiheads = 3,
                        latent_self_attn_type = 'multihead',
                        det_self_attn_type = 'multihead',
                        det_cross_attn_type = 'multihead',
                        isnorm = True,
                        use_self_attn=False,
                        use_deterministic_enc=True,
                        context_in_target = False
                       )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def prediction(lat, lon, data='N',do_eval=True):
    """Run model on test/val data"""
    path = 'data/Cleaned'+data+".csv"
    context_data = pd.read_csv(path)
    # print(context_data.head())
    if data == 'N':
        path = "./chkpt/checkpointN.pt"
    if data == 'P':
        path = "./chkpt/checkpointP.pt"
    if data == 'OC':
        path = "./chkpt/checkpointOC.pt"
    if data == 'K':
        path = "./chkpt/checkpointK.pt"

    #load the model
    Regressor.load_state_dict(torch.load(path, map_location=device))

    # target_x = lat
    # target_y = lat
    if do_eval:
        Regressor.eval()
    with torch.no_grad():
        # print("in prediction")

        target_x = np.asarray([[lon, lat]])
        context_x, context_y = context_data.iloc[:,:2], context_data.iloc[:,2:]
        context_x = torch.from_numpy(context_x.values).float()[None, :].to(device)
        context_y = torch.from_numpy(context_y.values).float()[None, :].to(device)
        target_x = torch.from_numpy(target_x).float()[None, :].to(device)
        # target_y = torch.from_numpy(target_y.values).float()[None, :].to(device)
        y_pred, _, _ = Regressor.forward(context_x, context_y, target_x, training = False)
        y_pred = np.clip(y_pred.cpu().detach().numpy()[0,0], 0, 1)
        
        print(y_pred)
        return y_pred


prediction(lat=19.51, lon=-75.43, data='K')