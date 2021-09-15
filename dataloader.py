
from . import NutrientsDataset, collate_fns

def DataLoader(data_train, 
                data_test,
                num_context = 80,
                num_extra_target = 18,
                batch_size = 32,
                context_in_target = False):

    df_train = data_train.copy()
    df_test = data_test.copy()


    hparams = dict(num_context,
                num_extra_target,
                batch_size,
                context_in_target)
    train_df = NutrientsDataset(df_train,hparams['num_context'],hparams['num_extra_target'])

    train_loader = DataLoader(train_df,
                            batch_size=hparams['batch_size'],
                            shuffle = True,
                            collate_fn=collate_fns(
                                hparams['num_context'],hparams['num_extra_target'], True, hparams['context_in_target']))


    return train_loader