import datetime
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import anndata
import numpy as np
import h5py
import torch
import json
from torch.autograd import Variable

from scbasset.utils import *
from scbasset.model_class import ModelClass
from scbasset.config import Config

import scbasset.deeptopic_utils as utils
import shap


########################################################################
# path to input data

start_directory = '../../../'

cbust_dir = 'result/tfbanformer-TF_to_region_hvg-64_768_4_8_7_32_relu_1-122-'
cbust_type = 'train'

seq_len = 768
# type_data, file_name = 'multiome_example', 'pbmc_multiome'
type_data, file_name = 'TF_to_region', 'TF_to_region_hvg'
# type_data, file_name = 'TF_to_region', 'TF_to_region_marker_genes'

data_path = start_directory + 'data/' + type_data + '/processed/'

ad_file = data_path + file_name + '-' + str(seq_len) + '-ad.h5ad'
h5_file = data_path + file_name + '-' + str(seq_len) + '-train_val_test.h5'

########################################################################
# Load data

# read h5ad file
ad = anndata.read_h5ad(ad_file)

f = h5py.File(h5_file, 'r')
X = f['X'][:].astype('float32')
Y = f['Y'][:].astype('float32')

X = torch.FloatTensor(X)

n_TFs = Y.shape[1]
ic(n_TFs, Y.shape[0])

# Load trained model
ic(torch.cuda.is_available())
device = "cuda"
if "cuda" in device and not torch.cuda.is_available():
    device = "cpu"
else:
    torch.cuda.set_device(0)

ic(device)

# Model config
config = Config()
config.h5_file = h5_file
config.bottleneck_size = 64
config.activation_fct = 'gelu'
config.batch_size = 32
# config.model_name = 'scbasset'
config.model_name = 'tfbanformer'
config.num_heads = 8
config.num_transforms = 7
config.repeat = 4
config.tower_multiplier = 1.122

print(config)

# load model
dashboard_model = ModelClass(config, n_TFs=n_TFs)
dashboard_model.activate_analysis()
dashboard_model.load_data(h5_file, shuffle=False)
# dashboard_model.load_weights(device, best=0, trained_model_dir='output/scbasset/TF_to_region_hvg/32_1344_6_TL/')

# ReLu not supported yet in deepexplainer (see comment in deepexplainer method)
config.activation_fct = "relu"
post_fix = str(config.bottleneck_size) + '_' + str(config.seq_length) + '_' + str(config.repeat)
post_fix = post_fix if config.model_name == 'scbasset' else post_fix + '_' + str(config.num_heads) + '_' + str(config.num_transforms)
post_fix += '_' + str(config.batch_size) + '_' + str(config.activation_fct) + '_' + str(config.tower_multiplier).replace('.', '-')


dashboard_model.load_weights(device, best=0, start_directory=start_directory, trained_model_dir='output/tfbanformer/TF_to_region_hvg/' + post_fix + '/')
dashboard_model.get_model_summary()
model = dashboard_model.model
model.to(device)

##############################################################
# Initialize deepexplainer

rn=np.random.choice(X.shape[0], 250, replace=False)
 
# Tensor = torch.cuda.FloatTensor
model.eval()
samples = X[rn]
# if torch.cuda.is_available():
#     samples = samples.type(Tensor)
device = "cpu"
model.to(device)
explainer = shap.DeepExplainer(model, Variable(samples))

path_dir = 'results/' + '-'.join(dashboard_model.model_path.split('/')[1:-1])
with open(path_dir + "/" + cbust_type + '_region_ISM_cbust.json', 'w') as json_file:
    cbust_data = json.load(json_file)
    print(cbust_data)

# post_fix = post_fix + '_' + 'TL'

data_type = '_test'
# Create the PdfPages object to which we will save the pages:
# The with statement makes sure that the PdfPages object is closed properly at
# the end of the block, even if an Exception occurs.
with PdfPages('results/in_silico_' + post_fix + data_type + '.pdf') as pdf:
    """examples 2 TFs"""
    # TF_list = [(724, 250)]
    # seq_list = [19, 176, 232, 293, 560]

    combinations = generate_combinations(cbust_data, ad)

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

        if len(combi) > 3: 
            TF_n, seq_id, TF_id, _ = combi
        else:
            TF_n, seq_id, TF_id = combi

        seq_onehot = X[seq_id:seq_id+1]
        region_id = ad.var.index[seq_id]
        ic(region_id)

        region_str = '{} ({})'.format(region_id, seq_id)

        TF_name = ad.obs['TF'][TF_id] + ' ' + ad.obs['cell_line'][TF_id] + ' ' + ad.obs['exp_id'][TF_id] + ' ' + ad.obs['motif_model'][TF_id] 
        ic(TF_name)

        # # Plot deepexplainer for the selected topics
        device = "cpu"
        model.to(device)
        torch.set_num_threads(16)
        axes[ax_id] = utils.plot_deepexplainer_givenax(explainer=explainer, fig=fig, ntrack=ntrack, 
                                            track_no=track_no1, seq_onehot=seq_onehot, 
                                            TF = TF_id, TF_name=TF_name, region_id=region_str)

        plot_cbust_on_ISM_chap(combi, cbust_data, axes[ax_id], region_id)

        device = "cuda"
        model.to(device)
        # Plot in silico mutagenesis for the selected topic
        ax = utils.plot_mutagenesis_givenax(model=model, fig=fig, ntrack=ntrack, track_no=track_no2, 
                                        seq_onehot=seq_onehot, num_classes=n_TFs, 
                                        TF = TF_id, TF_name=TF_name, region_id=region_str)

        plot_cbust_on_ISM_chap(combi, cbust_data, ax, region_id)

        if i%2 == 1:
            #to adjust y axis of two deepexplainer plot on the same region
            min_ = np.min([axes[0].get_ylim()[0],axes[1].get_ylim()[0] ])
            max_ = np.max([axes[0].get_ylim()[1],axes[1].get_ylim()[1] ])
            axes[0].set_ylim([min_, max_])
            axes[1].set_ylim([min_, max_])

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
