#!/usr/bin/env python
from scbasset.model_class import ModelClass, Config
import configargparse
from torch import cuda
from icecream import ic

import anndata
import h5py
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


def main():
    seq_len = 1344
    type = 'TF_to_region_ctx'
    h5_file = 'data/TF_to_region/processed/' + type + '-' + str(seq_len) + '-train_val_test.h5'

    config = Config()
    config.h5_file = h5_file
    config.bottleneck_size = 32
    config.activation_fct = 'gelu'
    # config.model_name = 'tfbanformer'
    config.model_name = 'scbasset'
    config.num_heads = 8
    config.num_transforms = 7
    config.repeat = 6
    config.cuda = 1
    config.batch_size = 64
    config.tower_multiplier = 1.222

    TL_type = 'region_accesibility'

    if config.model_name == 'tfbanformer':
        param = '{}_{}_{}_{}_{}_{}_{}_{}'.format(config.bottleneck_size, seq_len, config.repeat, config.num_heads, 
                                                 config.num_transforms, config.batch_size, config.activation_fct, 
                                                 str(config.tower_multiplier).replace('.', '-'))
    else:
        param = '{}_{}_{}_{}_{}_{}'.format(config.bottleneck_size, seq_len, config.repeat, config.batch_size,
                                           config.activation_fct, str(config.tower_multiplier).replace('.', '-'))                                         
    config.weights = 'output/' + config.model_name + '/{}/{}/'.format(TL_type, param)

    print(config)

    cuda.set_device(config.cuda)
    ic(cuda.current_device())

    TL_epochs = 50
    learning_rate = 0.01
    epochs_fine = 150
    learning_rate_fine = 0.0002

    dashboard = ModelClass(config=config, n_TFs=9409)
    transfer_learning = False if config.weights is None else True
    dashboard.activate_training(transfer_learning)
    dashboard.get_model_summary()
    
    if transfer_learning:
        dashboard.feature_extract(device='cuda', epochs=TL_epochs, best=0, start_directory='', learning_rate=learning_rate)
    else: 
        dashboard.fit()

    f = h5py.File(config.h5_file, 'r')
    X = f['X'][:].astype('float32')
    Y = f['Y'][:].astype('float32')

    n_TFs = Y.shape[1]
    ic(n_TFs, Y.shape[0])

    train_ids, val_ids, test_ids = f['train_ids'][:], f['val_ids'][:], f['test_ids'][:]

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X[train_ids][:5000]), torch.FloatTensor(Y[train_ids][:5000]))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    df_val_pred, df_val_y = dashboard.predict_batch(dataloader)
    df_val_scores = dashboard.contruct_auc_scores_by_TF(df_val_y, df_val_pred)

    print('auc roc score : ', df_val_scores['auc_roc'].mean())
    print('auc pr score : ', df_val_scores['auc_pr'].mean())

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X[val_ids]), torch.FloatTensor(Y[val_ids]))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    df_val_pred, df_val_y = dashboard.predict_batch(dataloader)
    df_val_scores = dashboard.contruct_auc_scores_by_TF(df_val_y, df_val_pred)

    print('auc roc score : ', df_val_scores['auc_roc'].mean())
    print('auc pr score : ', df_val_scores['auc_pr'].mean())

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X[test_ids]), torch.FloatTensor(Y[test_ids]))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    df_val_pred, df_val_y = dashboard.predict_batch(dataloader)
    df_val_scores = dashboard.contruct_auc_scores_by_TF(df_val_y, df_val_pred)

    print('auc roc score : ', df_val_scores['auc_roc'].mean())
    print('auc pr score : ', df_val_scores['auc_pr'].mean())

    ## Fine tuning
    weights = 'output/{}/{}/{}_TL/'.format(config.model_name, type, param)
    dashboard.fine_tuning(epochs=epochs_fine, start_epoch=TL_epochs, weights=weights, learning_rate=learning_rate_fine)

    f = h5py.File(config.h5_file, 'r')
    X = f['X'][:].astype('float32')
    Y = f['Y'][:].astype('float32')

    n_TFs = Y.shape[1]
    ic(n_TFs, Y.shape[0])

    train_ids, val_ids, test_ids = f['train_ids'][:], f['val_ids'][:], f['test_ids'][:]

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X[train_ids][:5000]), torch.FloatTensor(Y[train_ids][:5000]))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    df_val_pred, df_val_y = dashboard.predict_batch(dataloader)
    df_val_scores = dashboard.contruct_auc_scores_by_TF(df_val_y, df_val_pred)

    print('auc roc score : ', df_val_scores['auc_roc'].mean())
    print('auc pr score : ', df_val_scores['auc_pr'].mean())

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X[val_ids]), torch.FloatTensor(Y[val_ids]))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    df_val_pred, df_val_y = dashboard.predict_batch(dataloader)
    df_val_scores = dashboard.contruct_auc_scores_by_TF(df_val_y, df_val_pred)

    print('auc roc score : ', df_val_scores['auc_roc'].mean())
    print('auc pr score : ', df_val_scores['auc_pr'].mean())

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X[test_ids]), torch.FloatTensor(Y[test_ids]))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    df_val_pred, df_val_y = dashboard.predict_batch(dataloader)
    df_val_scores = dashboard.contruct_auc_scores_by_TF(df_val_y, df_val_pred)

    print('auc roc score : ', df_val_scores['auc_roc'].mean())
    print('auc pr score : ', df_val_scores['auc_pr'].mean())

if __name__ == "__main__":
    main()
