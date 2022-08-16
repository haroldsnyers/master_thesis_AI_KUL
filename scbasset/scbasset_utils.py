"""
key functions from (https://github.com/calico/scbasset)
used to build scBasset architecture. (modified)
"""

import anndata
import h5py
import numpy as np
import torch
from Bio import SeqIO
from icecream import ic
from scipy import sparse
from torch.autograd import Variable

from scbasset.basenji_utils import *
from scbasset.utils import get_intermediate_output


def make_seq_h5(
    input_bed,
    input_fasta,
    out_file,
    seq_len=1344,
):
    """Preprocess to generate h5 for scBasset training.
    Args:
        input_bed:      bed file. genomic range of peaks.
        input_fasta:    fasta file. genome fasta. (hg19, hg38, mm10 etc.)
        out_file:       output file name.
        seq_len:        peak size to train on. default to 1344.
    Returns:
        None.           Save a h5 file to 'out_file'. X: 1-hot encoded feature matrix.
    """
    # generate 1-hot-encoding from bed
    (seqs_dna, seqs_coords) = make_bed_seqs(
        input_bed,
        fasta_file=input_fasta,
        seq_len=seq_len,
    )
    dna_array = [dna_1hot(x) for x in seqs_dna]
    dna_array = np.array(dna_array)
    ids = np.arange(dna_array.shape[0])

    # save train_test_val splits
    f = h5py.File(out_file, "w")
    f.create_dataset(
        "X",
        data=dna_array,
        dtype="bool",
    )
    f.close()


def make_h5(
    input_ad,
    input_bed,
    input_fasta,
    out_file,
    seq_len=1344,
    train_ratio=0.9,
):
    """Preprocess to generate h5 for scBasset training.
    Args:
        input_ad:       anndata. the peak by cell matrix.
        input_bed:      bed file. genomic range of peaks.
        input_fasta:    fasta file. genome fasta. (hg19, hg38, mm10 etc.)
        out_file:       output file name.
        seq_len:        peak size to train on. default to 1344.
        train_ratio:    fraction of data used for training. default to 0.9.
    Returns:
        None.           Save a h5 file to 'out_file'. X: 1-hot encoded feature matrix.
                        Y: peak by cell matrix. train_ids: data indices used for train.
                        val_ids: data indices used for val. test_ids: data indices unused
                        during training, can be used for test.
    """
    # generate 1-hot-encoding from bed

    (seqs_dna, seqs_coords,) = make_bed_seqs(
        input_bed,
        fasta_file=input_fasta,
        seq_len=seq_len,
    )
    dna_array = [dna_1hot(x) for x in tqdm(seqs_dna, total=len(seqs_dna))]
    dna_array = np.array(dna_array)
    ids = np.arange(dna_array.shape[0])
    np.random.seed(10)
    test_val_ids = np.random.choice(
        ids,
        int(len(ids) * (1 - train_ratio)),
        replace=False,
    )
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(
        test_val_ids,
        int(len(test_val_ids) / 2),
        replace=False,
    )
    test_ids = np.setdiff1d(test_val_ids, val_ids)

    # generate binary peak*cell matrix
    ad = anndata.read_h5ad(input_ad)
    if sparse.issparse(ad.X):
        m = (np.array(ad.X.todense()).transpose() != 0) * 1
    else:
        m = (np.array(ad.X).transpose() != 0) * 1
        # m = np.array(ad.X)

    # save train_test_val splits
    f = h5py.File(out_file, "w")
    f.create_dataset(
        "X",
        data=dna_array,
        dtype="bool",
    )
    f.create_dataset("Y", data=m, dtype="int8")
    f.create_dataset(
        "train_ids",
        data=train_ids,
        dtype="int",
    )
    f.create_dataset(
        "val_ids",
        data=val_ids,
        dtype="int",
    )
    f.create_dataset(
        "test_ids",
        data=test_ids,
        dtype="int",
    )
    f.close()


def motif_score(tf, model, motif_fasta_folder, n_TFs):
    fasta_motif = "%s/shuffled_peaks_motifs/%s.fasta" % (
        motif_fasta_folder, tf)
    fasta_bg = "%s/shuffled_peaks.fasta" % motif_fasta_folder

    pred_motif = prediction_on_fasta(fasta_motif, model, n_TFs)
    pred_bg = prediction_on_fasta(fasta_bg, model, n_TFs)
    tf_score = pred_motif.mean(axis=0) - pred_bg.mean(axis=0)
    tf_score = (tf_score - tf_score.mean()) / tf_score.std()
    return tf_score


def prediction_on_fasta(fa, model, n_cells):
    records = list(SeqIO.parse(fa, "fasta"))
    seqs = [str(i.seq) for i in records]
    seqs_1hot = np.array([dna_1hot(i) for i in seqs])
    pred = imputation_Y_normalize(seqs_1hot, model)
    return pred

# perform imputation. Depth normalized.


def imputation_Y_normalize(X, model, scale_method=None):
    if torch.cuda.is_available():
        Y_pred = get_intermediate_output(model, model.nonLinear, X).cpu().numpy()
        w = model.final1.dense_layer.weight.detach().cpu().numpy().transpose()
    else:
        Y_pred = get_intermediate_output(model, model.nonLinear, X)[0]
        w = model.final1.dense_layer.weight.detach().transpose()
    Y_pred = Y_pred[:, None, :]
    accessibility_norm = np.dot(Y_pred.squeeze(), w)

    if scale_method == "all_positive":
        accessibility_norm = accessibility_norm - np.min(accessibility_norm)
    if scale_method == "sigmoid":
        accessibility_norm = np.divide(1, 1 + np.exp(-accessibility_norm),)

    return accessibility_norm



