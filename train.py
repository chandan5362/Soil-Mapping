from utils import split_data
import numpy as np
import torch

from model import  Regressor
from . import EarlyStopping
from val import validation


device = torch.device("cpu")
torch.cuda.empty_cache()




def train(data, train_loader,  num_epochs ):
    
    
    opt = torch.optim.Adam(Regressor.parameters(), lr = 1e-2, weight_decay = 1e-4)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    Regressor = Regressor.to(device)


    df_train, df_test = split_data(data)

    from tqdm.auto import tqdm 
    mse_loss_train = []
    mse_loss_eval = []
    elbo_loss_train  = []
    elbo_loss_eval = []
    mae_val_loss = []
    for epoch in range(num_epochs):
        loss = 0 
        mse_loss = 0
        Regressor.train()
        for batch in tqdm(train_loader):
            context_x, context_y, target_x, target_y = batch
            cx =context_x.to(device)
            cy = context_y.to(device)
            tx = target_x.to(device)
            ty = target_y.to(device)

            Regressor.zero_grad()

            _, losses, _ = Regressor.forward(cx, cy, tx, ty, training=True)
            losses['loss'].backward()
            loss += losses['loss'].cpu().detach().numpy()
            mse_loss+=losses['mse_loss'].cpu().detach().numpy()
            opt.step()
            
        loss /= len(train_loader)
        elbo_loss_train.append(loss)
        mse_loss_train.append(mse_loss/len(train_loader))
        print(epoch)
        print('ELBO train_loss', loss)
        print('RMSE train_loss', (mse_loss/len(train_loader))**0.5)
        
        yr, _, r2, losses_val, _ = validation(df_train, df_test)

        mse_loss_val = losses_val['mse_loss'].cpu().detach().numpy()
        mse_loss_eval.append(mse_loss_val)
        elbo_loss_val = losses_val['loss'].cpu().detach().numpy() 
        val_loss = np.mean(np.abs(yr))

        elbo_loss_eval.append(elbo_loss_val)
        mae_val_loss.append(val_loss)
        
        print('ELBO val_loss', elbo_loss_val)
        print('RMSE val_loss', mse_loss_val**0.5)
        print('MAE val_loss', val_loss)
        print('R2 score', r2)
    
        print("-----------------------------------------------------------------------")

        early_stopping(val_loss, Regressor)
        if early_stopping.counter==6:
            
            lr=opt.state_dict()['param_groups'][0]['lr']
            print('reducing lr {:2.2e} to {:2.2e}'.format(lr, lr/10))
            opt = torch.optim.Adam(Regressor.parameters(), lr=lr/10)

        if early_stopping.early_stop:
            print("Early stopping")
            break






