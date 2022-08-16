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
    #     'batch_size': 64, 'fct': 'relu', 'mult': 1.122, "TL": False},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 32, 'repeat':4, 'num_heads': 8, 'num_transforms': 11, 
    #     'batch_size': 64, 'fct': 'relu', 'mult': 1.222, "TL": False},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 1344, 'bottle': 64, 'repeat':4, 'num_heads': 7, 'num_transforms': 11, 
    #     'batch_size': 64, 'fct': 'relu', 'mult': 1.122, "TL": False},

    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 32, 'fct': 'relu', 'mult': 1.122, "TL": False},
    # {
    #     'file_name': 'TF_to_region_scplus_min5', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False},
    
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": False},
    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": False},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": True, 'best': 1},
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 0},
    
    # {
    #     'file_name': 'TF_to_region_ctx', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 0},

    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 1},
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 0},
    
    # {
    #     'file_name': 'TF_to_region_hvg', 'model': 'scbasset', 'seq_len': 1344, 'bottle': 32, 'repeat':6, 'residual': False, 'batch_size': 64, 
    #     'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 0},

    # {
    #     'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
    #     'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 1},
    {
        'file_name': 'TF_to_region_scplus', 'model': 'tfbanformer', 'seq_len': 768, 'bottle': 64, 'repeat':4, 'num_heads': 8, 'num_transforms': 7, 
        'batch_size': 64, 'fct': 'gelu', 'mult': 1.222, "TL": False, 'best': 1},
    
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
        # for h5 in ['TF_to_region_hvg-', 'TF_to_region_ctx-', 'TF_to_region_scplus-', 'region_accesibility-', 'TF_to_region_hvg_3k_min1-', 'TF_to_region_marker_genes_3k_min1-']:
        for cbust_type in ['train', 'test']:
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

            ad_file_tmp = data_path + 'TF_to_region_hvg' + '-' + str(seq_len) + '-ad.h5ad'
            h5_file_tmp = data_path + 'TF_to_region_hvg' + '-' + str(seq_len) + '-train_val_test.h5'

            # read h5ad file
            ad_tmp = anndata.read_h5ad(ad_file_tmp)

            f_tmp = h5py.File(h5_file_tmp, 'r')
            X_tmp = f_tmp['X'][:].astype('float32')
            Y_tmp = f_tmp['Y'][:].astype('float32')

            X_tmp = torch.FloatTensor(X_tmp)
            
            config = Config()
            config.model_name = param['model']
            config.h5_file = h5_file
            config.bottleneck_size = param['bottle']
            config.activation_fct = 'gelu'
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
            config.activation_fct = param['fct']
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
            ## Cbust

            # path_dir = 'results/' + '-'.join(model_path.split('/')[1:-1])

            # with open(path_dir + "/" + cbust_type + '_region_ISM.json') as json_file:
            #     data = json.load(json_file)
            #     print(data)

            # save_as_fasta_and_txt(data, ad, X, type_data=cbust_type, path_to_dir=path_dir)

            # path = r"cbust_bash.sh"
            # print(os.path.isfile(path))
            # with open(path, "r") as f:
            #     pass

            # for TF in data:
            #     print(TF)
            #     if len(TF.split('-')) > 1:
            #         TFs = TF.split('-')
            #         for i in range(len(TFs)):
            #             subprocess.call([path, 'fasta_' + data[TF]['TF'][i] + "_" + cbust_type + ".fa", 
            #                             path_dir, data[TF]['TF'][i] + "_" + cbust_type])
            #     else:
            #         subprocess.call([path, 'fasta_' + data[TF]['TF'] + "_" + cbust_type + ".fa", 
            #                         path_dir, data[TF]['TF'] + "_" + cbust_type])

            # motif_dict = extract_cbust_motifs(data, path_dir, cbust_type)

            # with open(path_dir + "/" + cbust_type + '_region_ISM_cbust.json', 'w') as json_file:
            #     json.dump(motif_dict, json_file)

            ##############################################################
            # Initialize deepexplainer

            rn=np.random.choice(X.shape[0], 250, replace=False)
            X = torch.FloatTensor(X)
            # Tensor = torch.cuda.FloatTensor
            model.eval()
            samples = X[rn]
            # if torch.cuda.is_available():
            #     samples = samples.type(Tensor)
            device = "cpu"
            model.to(device)
            explainer = shap.DeepExplainer(model, Variable(samples))

            with open('results/tfbanformer-TF_to_region_hvg-64_768_4_8_7_64_gelu_1-222' + "/" + cbust_type + '_region_ISM_cbust.json') as json_file:
                cbust_data = json.load(json_file)
                print(cbust_data)

            # Create the PdfPages object to which we will save the pages:
            # The with statement makes sure that the PdfPages object is closed properly at
            # the end of the block, even if an Exception occurs.
            with PdfPages('results/in_silico_' + '-'.join(model_path.split('/')[1:-1]) + "_best" + str(best) + "_" + cbust_type + '.pdf') as pdf:
                
                combinations = generate_combinations(cbust_data, ad_tmp)

                ntrack = 4
                axes = [None, None]
                for i, combi in enumerate(combinations):
                    ic(combi)
                    if i%2 == 0:
                        ax_id = 0
                        fig = plt.figure(figsize=(80,ntrack*5))
                        track_no1, track_no2 = 1, 2
                    else:
                        ax_id = 1
                        track_no1, track_no2 = 3, 4

                    if len(combi) > 5: 
                        TF_n, seq_id, TF_id, _, auc_roc, auc_pr = combi
                    else:
                        TF_n, seq_id, TF_id, auc_roc, auc_pr = combi

                    seq_onehot = X_tmp[seq_id:seq_id+1]
                    region_id = ad_tmp.var.index[seq_id]
                    ic(region_id)

                    region_str = '{} ({})'.format(region_id, seq_id)

                    try: 
                        TF_name = ad.obs['TF'][TF_id] + ' ' + ad.obs['cell_line'][TF_id] + ' ' + ad.obs['exp_id'][TF_id] + ' ' + ad.obs['motif_model'][TF_id] 
                    except:
                        TF_name = ad.obs['TF'][TF_id] + ' ' + ad.obs['motif_model'][TF_id] 

                    ic(TF_name)


                    # # Plot deepexplainer for the selected topics
                    device = "cpu"
                    model.to(device)
                    torch.set_num_threads(16)
                    axes[ax_id] = utils.plot_deepexplainer_givenax(explainer=explainer, fig=fig, ntrack=ntrack, 
                                                        track_no=track_no1, seq_onehot=seq_onehot, 
                                                        TF = TF_id, TF_name=TF_name, region_id=region_str)

                    title = "TF_" + str(TF_id) + ' : ' + TF_name + 'for sequence region : ' + region_str
                    title_add = '\n with auc_roc : ' + str("{:.4f}".format(auc_roc)) + ' and auc_pr : ' + str("{:.4f}".format(auc_pr))
                    
                    axes[ax_id].set_title(title+title_add)

                    try:
                        plot_cbust_on_ISM_chap(combi, cbust_data, axes[ax_id], region_id)
                    except:
                        pass

                    device = "cuda"
                    model.to(device)
                    # Plot in silico mutagenesis for the selected topic
                    ax = utils.plot_mutagenesis_givenax(model=model, fig=fig, ntrack=ntrack, track_no=track_no2, 
                                                    seq_onehot=seq_onehot, num_classes=n_TFs, 
                                                    TF = TF_id)

                    try:
                        plot_cbust_on_ISM_chap(combi, cbust_data, ax, region_id)
                    except:
                        pass

                    if i%2 == 1:
                        #to adjust y axis of two deepexplainer plot on the same region
                        # min_ = np.min([axes[0].get_ylim()[0],axes[1].get_ylim()[0] ])
                        # max_ = np.max([axes[0].get_ylim()[1],axes[1].get_ylim()[1] ])
                        # axes[0].set_ylim([min_, max_])
                        # axes[1].set_ylim([min_, max_])

                        # plt.title('Page One')
                        pdf.savefig(fig)  # saves the current figure into a pdf page
                        plt.close()

                # We can also set the file's metadata via the PdfPages object:
                d = pdf.infodict()
                d['Title'] = 'Multipage PDF Example'
                d['Author'] = 'Jouni K. Sepp\xe4nen'
                d['Subject'] = 'How to create a multipage pdf file and set its metadata'
                d['Keywords'] = 'PdfPages multipage keywords author title subject'
                d['CreationDate'] = datetime.datetime(2009, 11, 13)
                d['ModDate'] = datetime.datetime.today()

if __name__ == "__main__":
    main()