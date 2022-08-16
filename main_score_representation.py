#!/usr/bin/env python
import anndata
import numpy as np
import subprocess
import h5py
import torch
import json
import shap
import configargparse
import datetime
import seaborn as sns

from torch.autograd import Variable
from icecream import ic
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from scbasset.utils import *
from scbasset.model_class import ModelClass
from scbasset.config import Config

import scbasset.deeptopic_utils as utils


PARAMETERS_CONFIG_CBUST = [
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'relu', 'mult': 1.122},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 32, 'repeat':4, 'num_heads': 8, 'num_transforms': 11, 
    #     'batch_size': 64, 'fct': 'relu', 'mult': 1.222},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 1344, 'bottle': 64, 'repeat':4, 'num_heads': 7, 'num_transforms': 11, 
    #     'batch_size': 64, 'fct': 'relu', 'mult': 1.122},

    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 32, 'fct': 'relu', 'mult': 1.122},
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False},
    
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222},

    {
        'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
        'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 1},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 0},
    
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 1},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 0},

    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False},
    
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": False},

    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 1},

    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 0},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 1},

    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 0},
    
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 0},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 0},
]


def make_parser():
    parser = configargparse.ArgParser(
        description="train scBasset on scATAC data")
    parser.add_argument('--cuda', type=int, default=2,
                        help='CUDA device number, Default to 2')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    # Load trained model
    ic(torch.cuda.is_available())
    device = "cuda"
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"
    else:
        torch.cuda.set_device(args.cuda)
        
    ic(torch.cuda.current_device())


    for param in PARAMETERS_CONFIG_CBUST:
        ic(param)
        file_name = param['file_name']

        seq_len = str(param['seq_len'])
        data_path = 'data/TF_to_region/processed/'

        ad_file = data_path + file_name + '-' + str(seq_len) + '-ad.h5ad'
        h5_file = data_path + file_name + '-' + str(seq_len) + '-train_val_test.h5'

        # read h5ad file
        ad = anndata.read_h5ad(ad_file)

        f = h5py.File(h5_file, 'r')
        X = f['X'][:].astype('float32')
        Y = f['Y'][:].astype('float32')

        n_TFs = Y.shape[1]
        ic(n_TFs, Y.shape[0])

        
        config = Config()
        config.model_name = param['model']
        config.h5_file = h5_file
        config.bottleneck_size = param['bottle']
        config.activation_fct = param['fct']
        config.repeat = param['repeat']
        config.batch_size = param['batch_size']
        config.tower_multiplier = param['mult']
        if config.model_name == 'tfbanformer':
            config.num_heads = param['num_heads']
            config.num_transforms = param['num_transforms']
        if config.model_name == 'scbasset':
            config.residual_model = param['residual']

        # load model
        dashboard_model = ModelClass(config, n_TFs=n_TFs)
        dashboard_model.activate_analysis()
        dashboard_model.load_data(h5_file, shuffle=False)
        # dashboard_model.load_weights(device, best=0, trained_model_dir='output/scbasset/TF_to_region_hvg/32_1344_6_TL/')

        # ReLu not supported yet in deepexplainer (see comment in deepexplainer method)
        
        post_fix = str(config.bottleneck_size) + '_' + str(config.seq_length) + '_' + str(config.repeat)
        post_fix = post_fix if config.model_name == 'scbasset' else post_fix + '_' + str(config.num_heads) + '_' + str(config.num_transforms)
        post_fix += '_' + str(config.batch_size) + '_' + str(config.activation_fct) + '_' + str(config.tower_multiplier).replace('.', '-')

        if param['TL']:
            post_fix = post_fix + '_TL'

        model_path = 'output/' + param['model'] + '/' + file_name + '/' + post_fix + '/'

        best = param['best']

        dashboard_model.load_weights(device, best=best, start_directory='', trained_model_dir=model_path)
        dashboard_model.get_model_summary()
        model = dashboard_model.model
        model.to(device)

        ########################################################################
        # Score region representation

        latent_representation, weights = get_latent_representation_and_weights(model, X, Y)
        ic(latent_representation.shape, weights.shape)

        proj = get_TF_embedding(model) 
        print(len(proj))
        pd.DataFrame(proj).to_csv('results/projection_atac.csv')
        ad.obsm['projection'] = pd.read_csv('results/projection_atac.csv', index_col=0).values

        ad_regions = ad.T
        ad_regions = prepare_leiden_representation(adata=ad_regions, resolution=0.05)
        # sc.pp.neighbors(ad.T, use_rep='projection')
        # sc.tl.umap(ad)

        ad_latent = sc.AnnData(latent_representation)
        ad_latent = ad_latent

        ad_latent.obs.index = ad_regions.obs.index
        ad_latent = prepare_leiden_representation(adata=ad_latent, resolution=1)
        ad_latent.obs['leiden_original'] = ad_regions.obs['leiden'].values
        ad_regions.obs['leiden_learned'] = ad_latent.obs['leiden'].values

        sc.pp.filter_cells(ad_latent, min_genes=0)
        sc.pp.filter_genes(ad_latent, min_cells=0)


        # Compute jaccard index for latent representation
        df_jaccard_matrix_latent = compute_jaccard_matrix(ad_latent)

        ##############################################################
        # Score TF representation

        ad_TF = ad

        # sc.pp.neighbors(ad_TF, use_rep='projection')
        ad_TF = prepare_leiden_representation(ad_TF, resolution=3)

        ad_weights = sc.AnnData(weights)
        ad_weights.obs.index = ad_TF.obs.index
        ad_weights = prepare_leiden_representation(ad_weights, resolution=4)
        ad_weights.obs['leiden_original'] = ad_TF.obs['leiden'].values
        ad_TF.obs['leiden_learned'] = ad_weights.obs['leiden'].values

        data = pd.DataFrame(ad_weights.X)

        # Compute jaccard index for TF representation
        df_jaccard_matrix_TF = compute_jaccard_matrix(ad_weights)

        # Create the PdfPages object to which we will save the pages:
        # The with statement makes sure that the PdfPages object is closed properly at
        # the end of the block, even if an Exception occurs.

        with PdfPages('results/score_representation_' + '-'.join(dashboard_model.model_path.split('/')[1:3]) + '-' + post_fix + "_best" + str(best) + '.pdf') as pdf:
            
            print('starting pdf generation')
            fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(10, 15))
            sc.pl.umap(ad_regions, color='leiden', ax=axs[0], show=False)
            sc.pl.umap(ad_regions, color='leiden_learned', ax=axs[1], show=False)
            sc.pl.umap(ad_latent, color='leiden', ax=axs[3], show=False)

            try:
                sc.pl.umap(ad_regions, color='EXP030880.CD4_T-cells.CTCF.MA0139.1', ax=axs[2], show=False)
            except:
                sc.pl.umap(ad_latent, color='leiden_original', ax=axs[2])
            fig.tight_layout()
            pdf.savefig(fig)  # saves the current figure into a pdf page
            plt.close()

            plt.rcParams['text.usetex'] = False
            fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(10, 15))
            sc.pl.umap(ad_TF, color='leiden', ax=axs[0], show=False)
            sc.pl.umap(ad_TF, color='leiden_learned', ax=axs[1], show=False)
            # sc.pl.umap(ad_TF, color=TF_act + '_activity', ax=axs[1], cmap='coolwarm', vmin=-2, vmax=2, show=False)
            sc.pl.umap(ad_weights, color='leiden', ax=axs[2], show=False)
            sc.pl.umap(ad_weights, color='leiden_original', ax=axs[3])
            # sc.pl.draw_graph(ad_weights, ax=axs[3])
            fig.tight_layout()
            pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
            plt.close()

            # if LaTeX is not installed or error caught, change to `False`
            plt.rcParams['text.usetex'] = False
            fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))
            sns.heatmap(df_jaccard_matrix_latent, ax=axs[0])
            axs[0].set_title('Jaccard index for region clustering original vs learned representation')
            sns.heatmap(df_jaccard_matrix_TF, ax=axs[1])
            axs[1].set_title('Jaccard index for TF clustering original vs learned representation')

            pdf.savefig(fig)
            plt.close()

            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'Score representation'
            d['Author'] = 'Snyers H.'
            d['Subject'] = 'Score representation of region and tfs'
            d['Keywords'] = 'TF, representation'
            d['CreationDate'] = datetime.datetime(2022, 4, 27)
            d['ModDate'] = datetime.datetime.today()
            print('End pdf generation')

if __name__ == "__main__":
    main()