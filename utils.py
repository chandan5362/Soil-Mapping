import numpy as np
import torch

import seaborn as sns
from scipy.special import inv_boxcox
from scipy.stats import boxcox


def collate_fns(max_num_context, max_num_extra_target, sample, sort=True, context_in_target=True):
    def collate_fn(batch, sample=sample):
        x = np.stack([x for x, y in batch], 0)
        y = np.stack([y for x, y in batch], 0)

        # Sample a subset of random size
        num_context = np.random.randint(50, max_num_context)
        num_extra_target = np.random.randint(4, max_num_extra_target)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        
        x_context = x[:, :max_num_context]
        y_context = y[:, :max_num_context]
    
        x_target_extra = x[:, max_num_context:]
        y_target_extra = y[:, max_num_context:]
        
        if sample:

            x_context, y_context = npsample_batch(
                x_context, y_context, size=num_context
            )

            x_target_extra, y_target_extra = npsample_batch(
                x_target_extra, y_target_extra, size=num_extra_target, sort=sort
            )

        # do we want to compute loss over context+target_extra, or focus in on only target_extra?
        if context_in_target:
            x_target = torch.cat([x_context, x_target_extra], 1)
            y_target = torch.cat([y_context, y_target_extra], 1)
        else:
            x_target = x_target_extra
            y_target = y_target_extra

        
        return x_context, y_context, x_target, y_target

    return collate_fn



def npsample_batch(x, y, size=None, sort=False):
    
    """Sample datapoints from numpy arrays along 2nd dim."""
    inds = np.random.choice(range(x.shape[1]), size=size, replace=False)
    return x[:, inds], y[:, inds]


def split_data(data):
    """split the data into train and val"""

    # reduce the skewness of the data
    data_exp = data.copy()
    data_exp.drop(data[data['N'] == 0.0].index, inplace=True)
    val, lambda_ =  boxcox(data_exp['N'].values)
    data_exp['N'] = val

    data = data_exp[['N', 'lat', 'lon']].copy()
    #split dataset into train and test
    # split the dataset into train and test dataset
    ix = np.random.choice(data.shape[0],int(data.shape[0]*0.2),replace = False)
    data_train = data.iloc[[int(i) for i in range(data.shape[0]) if i not in ix]].reset_index(drop = True)
    data_test = data.iloc[ix].reset_index(drop = True)

    return data_train, data_test