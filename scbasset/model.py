from re import A, X
import re
from tkinter import N
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix
from torch import nn
from torch.autograd import Variable
from torch.nn import init
from icecream import ic


class LossFunctions:
    def bce_loss(self, real, predicted):
        loss = F.binary_cross_entropy_with_logits(
            predicted, real, reduction='none').mean()

        return loss


class GELU(nn.Module):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def forward(self, x):
        return torch.sigmoid(torch.tensor(1.702) * x) * x


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot):
        if self.training:
            rc_seq_1hot = torch.gather(seq_1hot, index=torch.tile(torch.LongTensor(
                [3, 2, 1, 0]).cuda(), ([seq_1hot.size(0), seq_1hot.size(1), 1])), dim=-1)
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[1])
            reverse_bool = torch.rand(1) > 0.5
            if reverse_bool:
                src_seq_1hot = rc_seq_1hot
            else:
                src_seq_1hot = seq_1hot

            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, torch.tensor(False)


class SwitchReverse(nn.Module):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self):
        super(SwitchReverse, self).__init__()

    def forward(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.size())
        if xd == 2:
            rev_axes = [0]
        elif xd == 3:
            rev_axes = [0, 1]
        else:
            raise ValueError(
                "Cannot recognize SwitchReverse input dimensions %d." % xd)

        if reverse:
            x = torch.flip(x, dims=rev_axes)

        return x


class StochasticShift(nn.Module):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad="uniform"):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.FloatTensor(1).uniform_(
                0, len(self.augment_shifts)).type(torch.int64)
            shift = torch.gather(self.augment_shifts, index=shift_i, dim=0)
            if shift:
                sseq_1hot = shift_sequence(seq_1hot, shift)
            else:
                sseq_1hot = seq_1hot

            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shift_max": self.shift_max, "pad": self.pad})
        return config


def shift_sequence(seq, shift, pad_value=0.25):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if len(seq.size()) != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.size()

    pad = pad_value * torch.ones_like(seq[:, 0: torch.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return torch.cat([pad, sliced_seq], dim=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return torch.cat([sliced_seq, pad], dim=1)

    if shift > 0:
        sseq = _shift_right(seq)
    else:
        sseq = _shift_left(seq)

    sseq = sseq.reshape(input_shape)

    return sseq


class ConvBlock(nn.Module):
    def __init__(self, x_dim, ch_dim, activation, n_filters=None, kernel_size=1, strides=1, dilation_rate=1, 
                 dropout=0, residual=False, pool_size=1, batch_norm=True, bn_momentum=0.90, padding='same'):
        """
            n_filters:       Conv1D filter number
            kernel_size:   Conv1D kernel_size
            activation     Activation function
            strides:       Conv1D strides
            dilation_rate: Conv1D dilation rate
            dropout:       Dropout rate probability
            residual:      Residual connection boolean
            pool_size:     Max pool width
            batch_norm:    Apply batch normalization
            bn_momentum:   BatchNorm momentum
        """
        super(ConvBlock, self).__init__()
        self.nonLinear = activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual
        self.pool_size = pool_size
        self.padding = padding
        self.dilation_rate = dilation_rate
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(n_filters, momentum=bn_momentum)
            if residual:
                init.zeros_(self.bn_layer.weight)
            else:
                init.ones_(self.bn_layer.weight)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_layer = nn.Conv1d(in_channels=ch_dim,
                                    out_channels=n_filters,
                                    kernel_size=kernel_size,
                                    stride=strides,
                                    padding='same',
                                    dilation=dilation_rate,
                                    bias=False
                                    )

        if residual:
            self.residual_layer = nn.Conv1d(
                ch_dim, n_filters, kernel_size=1)

        init.kaiming_normal_(self.conv_layer.weight, mode='fan_out')
        if pool_size > 1:
            if padding == "same":
                P = int(((strides-1)*x_dim-strides+pool_size)/2)
                self.maxpool_layer = nn.MaxPool1d(pool_size, padding=P)

    def forward(self, inputs):
        current = self.nonLinear(inputs)
        current = self.conv_layer(current)

        # batch norm
        if self.batch_norm:
            current = self.bn_layer(current)

        # dropout
        if self.dropout > 0:
            current = self.dropout_layer(current)

        # residual add
        if self.residual:
            inputs = self.residual_layer(inputs)
            current += inputs

        # pool
        if self.pool_size > 1:
            current = self.maxpool_layer(current)

        return current


class ConvTower(nn.Module):
    def __init__(self, x_dim_init, n_filters_init, n_filters_end=None, n_filters_mult=None, divisible_by=1, repeat=1, 
                 activation=GELU(), **kwargs):
        """Construct a reducing convolution block.
        Args:
            n_filters_init:  Initial Conv1D filters
            n_filters_end:   End Conv1D filters
            n_filters_mult:  Multiplier for Conv1D filters
            divisible_by:  Round filters to be divisible by (eg a power of two)
            repeat:        Tower repetitions
        Returns:
            [batch_size, seq_length, features] output sequence
        """
        super(ConvTower, self).__init__()
        self.conv_blocks = nn.ModuleList()

        def _round(x):
            return int(np.round(x / divisible_by) * divisible_by)

        # determine multiplier
        if n_filters_mult is None:
            assert n_filters_end is not None
            n_filters_mult = np.exp(
                np.log(n_filters_end / n_filters_init) / (repeat - 1))

        # initialize filters

        rep_filters_in = rep_filters_out = n_filters_init
        x_dim = x_dim_init

        for ri in range(repeat):
            # convolution
            self.conv_blocks.append(ConvBlock(x_dim=x_dim, ch_dim=_round(
                rep_filters_in), n_filters=_round(rep_filters_out), activation=activation, **kwargs))

            # update filters
            rep_filters_in = rep_filters_out
            rep_filters_out *= n_filters_mult
            x_dim = x_dim // 2

    def forward(self, inputs):
        current = inputs
        for layer in self.conv_blocks:
            current = layer(current)

        return current


class DenseBlock(nn.Module):
    def __init__(self, x_dim, ch_dim, activation, n_units=None, flatten=False, dropout=0, residual=False, batch_norm=True, bn_momentum=0.90):
        """Construct a single dense block.
           Args:
               n_units:        Numbe of Conv1D filters
               activation:     relu/gelu/etc
               flatten:        Flatten across positional axis
               dropout:        Dropout rate probability
               residual:       Residual connection boolean
               batch_norm:     Apply batch normalization
               bn_momentum:    BatchNorm momentum
           Returns:
               [batch_size, seq_length(?), features] output sequence
        """
        super(DenseBlock, self).__init__()
        self.nonLinear = activation
        self.flatten = flatten
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual
        # if n_units is None:
        #     n_units = inputs.shape[-1]
        self.dense_layer = nn.Linear(
            x_dim*ch_dim, n_units, bias=(not batch_norm))
        init.kaiming_normal_(self.dense_layer.weight, mode='fan_out')
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(n_units, momentum=bn_momentum)
            if residual:
                init.zeros_(self.bn_layer.weight)
            else:
                init.ones_(self.bn_layer.weight)
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, inputs):
        current = self.nonLinear(inputs)

        if self.flatten:
            current = torch.flatten(current, start_dim=1)

        # dense
        current = self.dense_layer(current)

        # batch norm
        if self.batch_norm:
            current = self.bn_layer(current)

        # dropout
        if self.dropout > 0:
            current = self.dropout_layer(current)

        # residual add
        if self.residual:
            current = inputs + current

        return current


class Final(nn.Module):
    def __init__(self, x_dim, ch_dim, n_units=None, flatten=False):
        """Final simple transformation before comparison to targets.
        Args:
            inputs:         [batch_size, seq_length, features] input sequence
            units:          Dense units
            activation:     relu/gelu/etc
            flatten:        Flatten positional axis.
        Returns:
            [batch_size, seq_length(?), units] output sequence
        """
        super(Final, self).__init__()
        self.dense_layer = nn.Linear(ch_dim, n_units, bias=True)
        init.kaiming_normal_(self.dense_layer.weight, mode='fan_out')
        # self.nonLinear = activation
        self.flatten = flatten

    def forward(self, inputs):
        
        current = inputs
        if self.flatten:
            current = torch.flatten(current, start_dim=1)

        current = self.dense_layer(current)
        #current = self.nonLinear(current)

        return current, self.dense_layer.weight


class scBasset(nn.Module):
    def __init__(self, bottleneck_size, n_TFs, seq_len=1344, residual=False, activation=GELU(), repeat=6, tower_multiplier=1.122):
        """create keras CNN model.
        Args:
            bottleneck_size:int. size of the bottleneck layer.
            n_cells:        int. number of cells in the dataset. Defined the number of tasks.
            seq_len:        int. peak size used to train. Default to 1344.
        """
        super(scBasset, self).__init__()
        self.conv_block1 = ConvBlock(
            x_dim=seq_len, ch_dim=4, n_filters=288, activation=activation, kernel_size=17, pool_size=3)

        n_filt_init, n_filt_mult = 288, tower_multiplier
        self.conv_tower1 = ConvTower(x_dim_init=448, n_filters_init=n_filt_init, n_filters_mult=n_filt_mult, repeat=repeat, kernel_size=5, 
                                     pool_size=2, residual=residual, activation=activation)
        ch_dim = round(n_filt_init*(n_filt_mult**(repeat-1)))                         
        n_filters = 256
        x_dim = int(seq_len / n_filters / 0.75)
        ic(ch_dim, n_filters, x_dim)
        self.conv_block2 = ConvBlock(
            x_dim=x_dim, ch_dim=ch_dim, n_filters=n_filters, activation=activation, kernel_size=1)

        self.dense_block1 = DenseBlock(x_dim=x_dim, ch_dim=256, flatten=True, activation=activation, n_units=bottleneck_size, 
                                        dropout=0.2,)
        self.nonLinear = activation
        self.final1 = Final(x_dim=1, ch_dim=bottleneck_size, n_units=n_TFs)
        
        self.StochasticReverseComplement_layer = StochasticReverseComplement()
        self.SwitchReverse_layer = SwitchReverse()
        self.StochasticShift_layer = StochasticShift(3)
        self.loss = LossFunctions()

    def forward(self, sequence):
        current = sequence
        (current, reverse_bool) = self.StochasticReverseComplement_layer(
            current)  # enable random rv
        current = self.StochasticShift_layer(current)  # enable random shift
        current = current.swapaxes(1, 2)
        current = self.conv_block1(current)
        current = self.conv_tower1(current)
        current = self.conv_block2(current)
        #current = current.swapaxes(1,2)
        current = self.dense_block1(current)
        # latent peak representation (size h)
        current = self.nonLinear(current)
        # z_TF : TF representation
        current, z_TF = self.final1(current)
        current = self.SwitchReverse_layer([current, reverse_bool])
        current = torch.flatten(current, start_dim=1)

        return current
