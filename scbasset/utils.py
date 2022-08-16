from typing import Union

import numpy as np
import pandas as pd
import scanpy as sc
import os
import torch

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread
from itertools import compress

from icecream import ic
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm, trange

from scbasset.basenji_utils import *
from scbasset.model import scBasset
from scbasset.tfbanformer import TfBanformer


def get_TF_embedding(model):
    """get TF embeddings from final layer weights of trained model"""
    # return model.final1.dense_layer.weight.transpose()
    # return model.layers[-3].get_weights()[0].transpose()
    # with pytorch no need to transpose the weight matrix as it is already in the correct direction
    if torch.cuda.is_available():
        return model.final1.dense_layer.weight.detach().cpu().numpy()
    else:
        return model.final1.dense_layer.weight


def get_intercept(model):
    """get intercept from trained model"""
    # get weight from tf returns in the case of a dense layer a list of two values, the kernel matrix and bias vector
    # In Pytorch the the kernel matrix can be retrieved with dense_layer.weigth and the bias vector can be retrieved
    # with dense_layer.bias
    # return model.layers[-3].get_weights()[1]
    return model.final1.dense_layer.bias.detach().numpy()


def get_latent_representation_and_weights(model: Union[scBasset, TfBanformer], X, Y):
    Tensor = torch.cuda.FloatTensor

    # build dataloaders
    data = TensorDataset(torch.FloatTensor(
        X), torch.FloatTensor(Y))
    dataloader = DataLoader(
        data, batch_size=128, shuffle=False, num_workers=0)

    with torch.no_grad():
        model.eval()
        latent, weights = [], []
        for i, data_batch in tqdm(enumerate(dataloader, 0), unit="batch", total=len(dataloader)):
            X, y = data_batch
            X = Variable(X.type(Tensor)) if torch.cuda.is_available() else Variable(X)
            # y = Variable(y.type(Tensor)) if torch.cuda.is_available() else Variable(y)
            
            latent_repr = get_intermediate_output(model, model.dense_block1, X)
            _, TF_repr = get_intermediate_output(model, model.final1, X)

            # pred.append(predictions)
            latent.append(latent_repr)
            if i == 0:
                if torch.cuda.is_available():
                    TF_repr = TF_repr.cpu().numpy()
                weights = TF_repr
            del latent_repr, TF_repr

    latent_representation = latent[0].cpu() if torch.cuda.is_available() else latent[0]
    for elem in latent[1:]:
        elem = elem.cpu().numpy() if torch.cuda.is_available() else elem
        latent_representation = np.concatenate(
            (latent_representation, elem), axis=0)

    return latent_representation, weights


def prepare_leiden_representation(adata, resolution=1):
    # function_list = [sc.pp.neighbors, sc.tl.umap, sc.tl.leiden, sc.tl.draw_graph]
    function_list = [sc.pp.neighbors, sc.tl.umap, sc.tl.leiden]
    for i, function in tqdm(enumerate(function_list), unit="function", total=len(function_list)):
        if i == 2:
            function(adata, resolution=resolution)
        else:
            function(adata)
    return adata


def compute_jaccard_matrix(ad_leiden):
    len_model = len(ad_leiden.obs.leiden.unique())
    len_orig = len(ad_leiden.obs.leiden_original.unique())
    jaccard_matrix_model = np.zeros((len_model, len_orig))
    for i in trange(len_model):
        for j in range(len_orig):
            model_bool = ad_leiden.obs.leiden == str(i)
            model = ad_leiden[model_bool].obs.index
            orig_bool = ad_leiden.obs.leiden_original == str(j)
            orig = ad_leiden[orig_bool].obs.index

            common = list(set(model).intersection(orig))
            jaccard_score_model = len(common)/len(set(model).union(orig))

            jaccard_matrix_model[i][j] = jaccard_score_model

    jaccard_matrix = compute_diagional_matrix(jaccard_matrix_model, len_model)

    return jaccard_matrix


def compute_diagional_matrix(matrix, col_lenght):
    jaccard_matrix_save = matrix.copy()
    for i, row in enumerate(matrix):
        max = np.max(row)
        for j, col in enumerate(row):
            if col != max:
                matrix[i, j] = 0

    jaccard_matrix_df = pd.DataFrame(matrix)
    jaccard_matrix_saved_df = pd.DataFrame(jaccard_matrix_save)

    rows = [i for i in range(col_lenght)]
    sorted = jaccard_matrix_df.sort_values(by=rows, ascending=False, axis=1)
    jaccard_matrix_saved_df = jaccard_matrix_saved_df[list(sorted.columns)]

    return jaccard_matrix_saved_df


def in_silico_mutagenis(sequence, model):
    return model(sequence)


Tensor = torch.cuda.FloatTensor


def get_intermediate_output(model, layer, x):
    features = {}
    layer.register_forward_hook(get_features(features, 'feats'))
    x = Variable(torch.cuda.FloatTensor(x).type(Tensor)) if torch.cuda.is_available() else Variable(torch.FloatTensor(x))
    out = model(x)
    return features['feats']


def get_features(features, name):
    "HELPER FUNCTION FOR FEATURE EXTRACTION"
    def hook(model, input, output):
        if len(output) == 2:
            features[name] = [output[0].detach(), output[1].detach()]
        else:
            features[name] = output.detach()
    return hook


### CBUST Preprocessing Functions

def extract_regions(y_pred, y_true, list_regions, df_scores):
    dict_regions_info = {}
    for elem in list_regions:
        if type(elem) is int:
            TF = df_scores['TF'][elem]
            tmp = pd.DataFrame(y_pred[elem])
            tmp[TF] = y_true[elem].values
            region_indexes = list(tmp.loc[(tmp[elem] == 1) & (tmp[TF] == 1)].index.values)

            region_list = random.sample(region_indexes, 20) if len(region_indexes) > 20 else region_indexes

            dict_regions_info[elem] = {}
            dict_regions_info[elem]['TF'] = TF
            dict_regions_info[elem]['motif_model'] = df_scores['motif_model'][elem]
            dict_regions_info[elem]['auc_roc'] = float(df_scores['auc_roc'][elem])
            dict_regions_info[elem]['auc_pr'] = float(df_scores['auc_pr'][elem])
            dict_regions_info[elem]['regions'] = [int(x) for x in region_list]

        else:
            TFs = [df_scores['TF'][elem[i]] for i in range(len(elem))]
            tmp = y_pred[elem]
            tmp[TFs] = y_true[elem]
            region_indexes = list(tmp.loc[(tmp[elem[0]] == 1) & (tmp[elem[1]] == 1) & (tmp[TFs[0]] == 1) & (tmp[TFs[1]] == 1)].index.values)
            
            region_list = random.sample(region_indexes, 20) if len(region_indexes) > 20 else region_indexes

            id = "-".join((str(elem[0]), str(elem[1])))
            dict_regions_info[id] = {}
            dict_regions_info[id]['TF'] = TFs
            
            dict_regions_info[id]['motif_model'] = (df_scores['motif_model'][elem[0]], df_scores['motif_model'][elem[1]])
            dict_regions_info[id]['auc_roc'] = (float(df_scores['auc_roc'][elem[0]]), df_scores['auc_roc'][elem[1]])
            dict_regions_info[id]['auc_pr'] = (float(df_scores['auc_pr'][elem[0]]), df_scores['auc_pr'][elem[1]])
            dict_regions_info[id]['regions'] = [int(x) for x in region_list]

    return dict_regions_info

DEFAULT_PAIR_ID = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

def decode_one_hot(sequence_list):
    decoded_seq_list = []
    for i in range(len(sequence_list)):
        sequence = sequence_list[i]
        string_seq = ''
        for i in range(len(sequence)):
            string_seq += DEFAULT_PAIR_ID[np.argmax(sequence[i])]
        decoded_seq_list.append(string_seq)
    return decoded_seq_list

def save_as_fasta_and_txt(data_dict, ad, X, path_to_dir, type_data='train'):
    for TF in data_dict:
        seq_list_ids = data_dict[TF]['regions']
        seq_list = X[seq_list_ids]
        region_lists = ad.var.index[seq_list_ids]

        seq_list_decoded = decode_one_hot(seq_list)
        if len(TF.split('-')) > 1:
            TFs = [int(x) for x in TF.split('-')]
            for i, TF_n in enumerate(TFs):
                save_TF_fasta_and_txt(data_dict[TF]['TF'][i], data_dict[TF]['motif_model'][i], region_lists, 
                                      seq_list_decoded, type_data, path_to_dir)
        else:
            save_TF_fasta_and_txt(data_dict[TF]['TF'], data_dict[TF]['motif_model'], 
                                  region_lists, seq_list_decoded, type_data, path_to_dir)

def save_TF_fasta_and_txt(TF_name, motif, region_lists, seq_list_decoded, type_data, path_to_dir):
    filename_prefix = TF_name + "_" + type_data
    fasta_file_test = open(path_to_dir + "/fasta_" + filename_prefix + ".fa", 'w')
    for i, seq in enumerate(seq_list_decoded):
        fasta_file_test.write('>' + region_lists[i] + '\n' + seq + "\n")
    fasta_file_test.close()

    with open(path_to_dir + "/selected_motifs_" + filename_prefix +'.txt', 'w') as f:
        motif = "jaspar__" + motif
        path = "/staging/leuven/stg_00002/lcb/icistarget/data/motifCollection/v9/singletons/"
        
        if not os.path.isfile(path + motif + ".cb"):
            for i in reversed(range(6)):
                motif = motif[:-1] + str(i)
                if os.path.isfile(path + motif + ".cb"):
                    break

        print(motif)
        f.write(motif)
        f.write('\n')

def extract_cbust_motifs(data, file_dir, cbust_type):
    motif_dict = {}
    
    for TF in data:
        if len(TF.split('-')) > 1:
            TFs_id = "-".join(data[TF]['TF'])
            motif_dict[TFs_id] = {}
            TFs = TF.split('-')
            for i, (TF_n, TF_id) in enumerate(zip(data[TF]['TF'], TFs)):
                gff_filename = file_dir + "/fasta_" + TF_n + "_" + cbust_type + ".selected.motif_" + TF_n + "_" + cbust_type + ".gff"
                selected_motifs_filename = file_dir + "/selected_motifs_" + TF_n + "_" + cbust_type + ".txt"

                seq_names, cbust_mot_dict = gff_to_npz(gff_filename) 
                all_motifs_dict = create_motifs_dict(motif_filename=selected_motifs_filename)
                motif_dict[TFs_id][TF_n] = {}
                motif_dict[TFs_id][TF_n]['cbust_mot'] = cbust_mot_dict
                motif_dict[TFs_id][TF_n]['all_mot'] = all_motifs_dict
                motif_dict[TFs_id][TF_n]['TF_info'] = {"TF_id": int(TF_id), 'auc_roc': data[TF]['auc_roc'][i], 'auc_pr': data[TF]['auc_pr'][i]}

        else:
            motif_dict[data[TF]['TF']] = {}
            gff_filename = file_dir + "/fasta_" + data[TF]['TF'] + "_" + cbust_type + ".selected.motif_" + data[TF]['TF'] + "_" + cbust_type + ".gff"
            selected_motifs_filename = file_dir + "/selected_motifs_" + data[TF]['TF'] + "_" + cbust_type + ".txt"

            seq_names, cbust_mot_dict = gff_to_npz(gff_filename) 
            all_motifs_dict = create_motifs_dict(motif_filename=selected_motifs_filename)

            motif_dict[data[TF]['TF']]['cbust_mot'] = cbust_mot_dict
            motif_dict[data[TF]['TF']]['all_mot'] = all_motifs_dict
            motif_dict[data[TF]['TF']]['TF_info'] = {"TF_id": TF, 'auc_roc': data[TF]['auc_roc'], 'auc_pr': data[TF]['auc_pr']}

    return motif_dict

def plot_cbust_on_ISM_chap(combi, cbust_data, ax, region):
    if len(combi) > 5:
        single_m = cbust_data[combi[3]][combi[0]]['cbust_mot'][region]
        all_motif = cbust_data[combi[3]][combi[0]]['all_mot']
    else:
        single_m = cbust_data[combi[0]]['cbust_mot'][region]
        all_motif = cbust_data[combi[0]]['all_mot']
    threshold, st, end = 0, 0, 768

    for motif in [xx for xx in all_motif if xx in list(single_m.keys())]: 

        color = all_motif[motif][0]  
            
        for single_motif in single_m[motif]:
            if single_motif[2] >= threshold:
                if single_motif[3]=='-':
                    ax.add_patch(matplotlib.patches.Rectangle(xy=[single_motif[0]-st,-1*single_motif[2]] ,
                                                        width=single_motif[1]-single_motif[0] ,
                                                        height=single_motif[2],
                                                        color=color, fill=False, linewidth=3))
                else:
                    ax.add_patch(matplotlib.patches.Rectangle(xy=[single_motif[0]-st,0] ,
                                                        width=single_motif[1]-single_motif[0] ,
                                                        height=single_motif[2],
                                                        color=color, fill=False, linewidth=3)) 

def generate_combinations(cbust_data, adata):
    combinations = []
    for TF in cbust_data:
        if len(TF.split('-')) > 1:
            TFs = TF.split('-')
            tmp = []
            for tf in TFs:
                print(cbust_data[TF][tf]['TF_info']['TF_id'])
                regions = []
                for region in cbust_data[TF][tf]['cbust_mot']:
                    t = region == adata.var_names
                    regions.append(list(compress(range(len(t)), t))[0])
                tmp.append(regions)

            common = list(set(tmp[0]) & set(tmp[1]))

            TF1_id = cbust_data[TF][TFs[0]]['TF_info']['TF_id']
            TF2_id = cbust_data[TF][TFs[1]]['TF_info']['TF_id']

            for reg in common:
                combinations.append([TFs[0], reg, TF1_id, TF, cbust_data[TF][TFs[0]]['TF_info']['auc_roc'], cbust_data[TF][TFs[0]]['TF_info']['auc_pr']])
                combinations.append([TFs[1], reg, TF2_id, TF, cbust_data[TF][TFs[1]]['TF_info']['auc_roc'], cbust_data[TF][TFs[1]]['TF_info']['auc_pr']])
            
            for reg in [x for x in tmp[0] if x not in common]:
                combinations.append([TFs[0], reg, TF1_id, TF, cbust_data[TF][TFs[0]]['TF_info']['auc_roc'], cbust_data[TF][TFs[0]]['TF_info']['auc_pr']])
                combinations.append([TFs[1], reg, TF2_id, TF, cbust_data[TF][TFs[1]]['TF_info']['auc_roc'], cbust_data[TF][TFs[1]]['TF_info']['auc_pr']])

            for reg in [x for x in tmp[1] if x not in common]:
                combinations.append([TFs[0], reg, TF1_id, TF, cbust_data[TF][TFs[0]]['TF_info']['auc_roc'], cbust_data[TF][TFs[0]]['TF_info']['auc_pr']])
                combinations.append([TFs[1], reg, TF2_id, TF, cbust_data[TF][TFs[1]]['TF_info']['auc_roc'], cbust_data[TF][TFs[1]]['TF_info']['auc_pr']])

        else:
            print(cbust_data[TF]['TF_info']['TF_id'])
            regions = []
            for region in cbust_data[TF]['cbust_mot']:
                t = region == adata.var_names
                combinations.append([TF, list(compress(range(len(t)), t))[0], int(cbust_data[TF]['TF_info']['TF_id']), 
                                     cbust_data[TF]['TF_info']['auc_roc'], cbust_data[TF]['TF_info']['auc_pr']])

    return combinations

#### Cbust processing functions (ITASK)
def gff_to_npz(filename):
    with open(filename) as file:
        seq_names = []
        main_dict = {}
        for line in file:
            if line.startswith("#"):
                continue
            tabs = line.strip().split('\t')
            seq_name = tabs[0]
            start = int(tabs[1])
            end = int(tabs[2])
            motif_name = tabs[3]
            score = float(tabs[4])
            strand = tabs[5]

            if seq_name not in main_dict:
                main_dict[seq_name]={}
                seq_names.append(seq_name)
            if motif_name not in main_dict[seq_name]:
                main_dict[seq_name][motif_name] = []
            main_dict[seq_name][motif_name].append([start,end,score,strand]) 
    return seq_names,main_dict

def plot_cbust_with_logo(plotting_id, all_motifs_dict, cbust_mot_dict, threshold=3,st=0,end=500, plot_logos=False, ymin=-10, ymax=10):
    fig = plt.figure(figsize=(40,3))
    if plotting_id in cbust_mot_dict:
        single_m = cbust_mot_dict[plotting_id]
        ax = fig.add_subplot(1,1,1)
        ax.set_title(plotting_id)
        ax.plot(np.zeros(end-st), color = 'gray')
        for motif in [xx for xx in all_motifs_dict if xx in list(single_m.keys())]: 
            color = all_motifs_dict[motif][0]           
            for single_motif in single_m[motif]:
                if single_motif[2] >= threshold:
                    if single_motif[3]=='-':
                        ax.add_patch(matplotlib.patches.Rectangle(xy=[single_motif[0]-st,-1*single_motif[2]] ,
                                                         width=single_motif[1]-single_motif[0] ,
                                                         height=single_motif[2],
                                                         color=color, fill=False, linewidth=3))
                    else:
                        ax.add_patch(matplotlib.patches.Rectangle(xy=[single_motif[0]-st,0] ,
                                                         width=single_motif[1]-single_motif[0] ,
                                                         height=single_motif[2],
                                                         color=color, fill=False, linewidth=3))    
        _ = ax.set_xticks(np.arange(0, end-st+1, 10))
        ax.axis([0,end-st+1,ymin,ymax])
    else:
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.zeros(end-st), color = 'gray')
        _ = ax.set_xticks(np.arange(0, end-st+1, 10))
        ax.axis([0,end-st+1,ymin,ymax])

    if plot_logos:
        number_of_motifs = len(all_motifs_dict)
        fig = figure(figsize=(5*number_of_motifs,4))
        for k,name in enumerate(all_motifs_dict):
            url = "http://motifcollections.aertslab.org/v9/logos/" + name + ".png"
            a=fig.add_subplot(1,number_of_motifs,k+1)   
            image = imread(url)
            imshow(image,cmap='Greys_r')
            #a.set_title(selected_mot_col[k],color=selected_mot_col[k].split('-')[0],fontweight="bold")
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)
            a.get_yaxis().set_ticks([])
            a.get_yaxis().set_ticks([])
            a.set_yticklabels([])
            a.set_xticklabels([])
            a.set_xlabel(name,color=all_motifs_dict[name][0],fontweight="bold")

COLOR = [ 
    "#004D43", "#4FC601", "#809693", "#008941", "#FF4A46", "#A30059",
    "#7A4900", "#B79762", "#0000A6", "#8FB0FF", "#FF34FF", "#997D87",
    "#5A0007", "#1CE6FF", "#000000", "#006FA6", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",]

def create_motifs_dict(motif_filename):
    all_motifs_dict = {}
    with open(motif_filename,'r') as rmot:
        counter=0
        for line in rmot:
            all_motifs_dict[line.strip().split('\t')[0].rstrip('.cb')] = [COLOR[counter]]
            counter+=1

    return all_motifs_dict