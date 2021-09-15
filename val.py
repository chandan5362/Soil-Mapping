
import torch
from sklearn.metrics import r2_score



def validation(Regressor, device, data_train, data_test, do_eval=True):
    """Run model on test/val data"""
    if do_eval:
        Regressor.eval()
    with torch.no_grad():
        target_x, target_y = data_test.iloc[:,1:], data_test.iloc[:,:1]
        context_x, context_y = data_train.iloc[:,1:], data_train.iloc[:,:1]
#         print(context_x.shape, context_y.shape)

        context_x = torch.from_numpy(context_x.values).float()[None, :].to(device)
        context_y = torch.from_numpy(context_y.values).float()[None, :].to(device)
        target_x = torch.from_numpy(target_x.values).float()[None, :].to(device)
        target_y = torch.from_numpy(target_y.values).float()[None, :].to(device)
        y_pred, losses, extra = Regressor.forward(context_x, context_y, target_x, target_y, training = False)

        yr=(target_y-y_pred)[0].detach().cpu().numpy()
        r2 = r2_score(target_y.detach().cpu().numpy().flatten(), y_pred.detach().cpu().numpy().flatten())
        return yr, y_pred, r2, losses, extra 