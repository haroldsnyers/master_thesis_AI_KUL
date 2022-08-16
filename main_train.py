#!/usr/bin/env python
from scbasset.model_class import ModelClass, Config
import configargparse
from torch import cuda
from icecream import ic


PARAMETERS_CONFIG = [
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 400},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 400},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 300},
    # {
    #     'file_name': 'region_accesibility', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 70},
    {
        'file_name': 'region_accesibility', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
        'num_transforms': 7, 'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 70, 'version': None},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 32, 'fct': 'gelu', 'mult': 1.222, 'epochs': 200, 'version': 'v1'},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 32, 'fct': 'gelu', 'mult': 1.222, 'epochs': 200, 'version': 'v2'},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 32, 'fct': 'gelu', 'mult': 1.222, 'epochs': 200, 'version': 'v3'},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 200, 'version': 'v4'},
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, 'epochs': 400, 'version': 'v5'},
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 64, 'fct': 'relu', 'mult': 1.222, 'epochs': 600},
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 32, 'fct': 'relu', 'mult': 1.222, 'epochs': 600},
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 
    #     'num_transforms': 7, 'batch_size': 32, 'fct': 'gelu', 'mult': 1.222, 'epochs': 600},
]

def make_parser():
    parser = configargparse.ArgParser(
        description="train scBasset on scATAC data")
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train. Default to 1000.')
    parser.add_argument('--cuda', type=int, default=2,
                        help='CUDA device number, Default to 2')
    parser.add_argument('--model', type=str, default='scbasset',
                        help='Which model to choose from, default to scbasset. Choice between scbasset and scbasset2')

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    cuda.set_device(args.cuda)
    ic(cuda.current_device())

    config = Config()

    for param in PARAMETERS_CONFIG:
        ic(param)
        config.model_name = param['model']
        config.epochs = param['epochs']

        h5 = param['file_name']
        # for h5 in ['TF_to_region_hvg-', 'TF_to_region_ctx-', 'TF_to_region_scplus-', 'TF_to_region_scplus_5min-', 'region_accesibility-', 'TF_to_region_hvg_3k_min1-', 'TF_to_region_marker_genes_3k_min1-', 'TF_to_region_hvg_grouped_orig-']:
        h5_file = 'data/TF_to_region/processed/' + h5 + '-' + str(param['seq_len']) + '-train_val_test.h5'

        config.h5_file = h5_file
        config.bottleneck_size = param['bottle']
        config.activation_fct = param['fct']
        config.repeat = param['repeat']
        config.batch_size = param['batch_size']
        config.tower_multiplier = param['mult']
        config.version = param['version']
        if config.model_name == 'tfbanformer':
            config.num_heads = param['num_heads']
            config.num_transforms = param['num_transforms']
        if config.model_name == 'scbasset':
            config.residual_model = param['residual']

        dashboard = ModelClass(config=config)
        dashboard.activate_training()
        dashboard.get_model_summary()
        dashboard.fit()

if __name__ == "__main__":
    main()
