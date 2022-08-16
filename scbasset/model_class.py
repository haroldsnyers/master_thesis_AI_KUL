
import attr
import os
import h5py
import numpy as np
import pandas as pd
import datetime

import torch
import torch.optim as optim

from torch import nn, FloatTensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from icecream import ic
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from scbasset.model import scBasset, GELU, Final
from scbasset.tfbanformer import TfBanformer
from scbasset.config import Config

@attr.s
class ModelClass:
    '''
    The Model class for initiliazing model for easier use
    :param config: config object to pass to class for initiliazing model
    '''

    config = attr.ib(type=Config)
    model = attr.ib(default=None)
    optimizer = attr.ib(default=None)
    train_dataloader = attr.ib(default=None)
    val_dataloader = attr.ib(default=None)
    activation = attr.ib(default=None)
    n_TFs = attr.ib(default=None)
    data_file = attr.ib(default=None)

    ACTIVATION_FUNCTIONS = {
        'gelu': GELU(),
        'relu': nn.ReLU()
    }

    def activate_training(self, transfer_learning=False):
        self.load_data(self.config.h5_file, transfer_learning)
        self.activate_analysis()
        self.initialize_writter_and_model_path()
        self.model.float().cuda()

        self._set_optimizer()

    def activate_analysis(self):
        self.set_model_path()
        self.set_activation_fct()
        self.set_model()

    def set_model(self):
        if self.config.model_name == 'scbasset':
            self.model = scBasset(self.config.bottleneck_size, self.n_TFs, seq_len=self.config.seq_length, 
                                  residual=self.config.residual_model, activation=self.activation, 
                                  tower_multiplier=self.config.tower_multiplier
                                  )
        else:
            self.model = TfBanformer(self.config.bottleneck_size, self.n_TFs, seq_len=self.config.seq_length, 
                                     residual=self.config.residual_model, activation=self.activation, repeat=self.config.repeat,
                                     num_heads=self.config.num_heads, num_blocks=self.config.num_transforms, 
                                     tower_multiplier=self.config.tower_multiplier
                                    )


    def set_activation_fct(self):
        self.activation = self.ACTIVATION_FUNCTIONS[self.config.activation_fct]

    def _set_optimizer(self, learning_rate=None):
        learning_rate_ = self.config.learning_rate if learning_rate is None else learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate_, betas=(0.95, 0.9995))
        # ic(self.optimizer)
        
    def get_model_summary(self):
        # ic(self.model)
        summary(self.model, input_size=[(self.config.batch_size, self.config.seq_length, 4)])
    
    def load_data(self, h5_file, transfer_learning=False, shuffle=True):
        f = h5py.File(h5_file, 'r')
        X = f['X'][:].astype('float32')
        Y = f['Y'][:].astype('float32')

        # Split train-validation set
        train_ids, val_ids, test_ids = f['train_ids'][:], f['val_ids'][:], f['test_ids'][:]

        X_train, Y_train = X[train_ids], Y[train_ids]
        X_val, Y_val = X[val_ids], Y[val_ids] 
        X_test, Y_test = X[test_ids], Y[test_ids] 

        if transfer_learning:
            self.n_TFs_TL = Y.shape[1]
        else:
            self.n_TFs = Y.shape[1]
        ic(self.n_TFs, len(X_train), len(Y_train))

        # build dataloaders
        train_data = TensorDataset(FloatTensor(X_train), FloatTensor(Y_train))
        self.train_dataloader = DataLoader(
            train_data, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=0)

        val_data = TensorDataset(FloatTensor(X_val), FloatTensor(Y_val))
        self.val_dataloader = DataLoader(
            val_data, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=0)

        test_data = TensorDataset(FloatTensor(X_test), FloatTensor(Y_test))
        self.test_dataloader = DataLoader(
            test_data, batch_size=self.config.batch_size, shuffle=shuffle, num_workers=0)

        features, labels = next(iter(self.train_dataloader))
        ic(features.shape)

    def set_model_path(self, h5_file=None):
        h5_file = self.config.h5_file if h5_file is None else h5_file

        self.data_file = h5_file.split("/")[-1].split("-")[0]
        if len(h5_file.split("/")[-1].split("-")) > 2:
            self.config.seq_length = int(h5_file.split("/")[-1].split("-")[1])
        
        residual_str = "_residual" if self.config.residual_model else ""

        self.model_path = self.config.out_dir + "/" + self.config.model_name + '/' + \
            self.data_file + residual_str + '/' + str(self.config.bottleneck_size) + "_" + \
            str(self.config.seq_length) + "_" + str(self.config.repeat)

        if self.config.model_name == 'tfbanformer':
            self.model_path = self.model_path + "_" + str(self.config.num_heads) + "_" + str(self.config.num_transforms)

        multiplier = str(self.config.tower_multiplier).replace('.', '-')

        self.model_path += '_' + str(self.config.batch_size) + "_" + self.config.activation_fct +  '_' + multiplier 
        
        if self.config.weights is not None:
            self.model_path = self.model_path + "_TL" 

        if self.config.version is not None:
            self.model_path = self.model_path + "_" + self.config.version
        
        self.model_path += "/"
        

    def initialize_writter_and_model_path(self):
        ic(self.data_file, self.config.seq_length, self.config.bottleneck_size)
        ic(self.config.epochs, self.config.batch_size, self.config.model_name, 
           self.config.residual_model, self.config.activation_fct, self.config.num_heads,
           self.config.repeat)

        residual_str = "_residual" if self.config.residual_model else ""

        summary_writter_path = self.config.logs + '/logs/' + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + \
            self.config.model_name + residual_str + "_" + self.data_file + "_" + \
            str(self.config.bottleneck_size) + "_" + str(self.config.seq_length) + "_" + str(self.config.repeat)

        if self.config.model_name == 'tfbanformer':
            summary_writter_path = summary_writter_path + "_" + str(self.config.num_heads) + "_" + str(self.config.num_transforms)

        multiplier = str(self.config.tower_multiplier).replace('.', '-')
        summary_writter_path += '_' + str(self.config.batch_size) + "_" + self.config.activation_fct +  '_' + multiplier

        if self.config.weights is not None:
            summary_writter_path = summary_writter_path + "_TL" 

        # Initialize tensorboard
        self.writer = SummaryWriter(summary_writter_path)

        if not os.path.exists(self.model_path):
            Path(self.model_path).mkdir(parents=True, exist_ok=True)

    def fit(self, epochs=None, start_epoch=0):
        # Training
        self.model.train()
        epochs_ = self.config.epochs if epochs is None else epochs 
        start_iteration = start_epoch * len(self.train_dataloader)
        ic(start_iteration)
        for epoch in range(epochs_):
            loss_l, auc_roc_l, auc_pr_l = [], [], []
            auc_roc_prev = 0
            auc_pr_prev = 0
            for i, data_batch in tqdm(enumerate(self.train_dataloader, 0), unit="batch", total=len(self.train_dataloader)):
                self.optimizer.zero_grad()

                predictions, label, loss = self.compute_pred_label_loss(data_batch)

                loss.backward()
                loss_l.append(loss.detach().item())

                self.optimizer.step()

                (auc_roc, auc_pr, 
                 recall_sc, precision_sc, 
                 accuracy_sc, f1_sc) = self.compute_metrics(label, predictions)

                self.write_metrics(start_epoch + epoch, loss.item(), auc_roc, auc_pr, recall_sc, 
                                   precision_sc, accuracy_sc, f1_sc, i, 'train')
                
                if auc_roc == 0:
                    auc_roc, auc_pr = auc_roc_prev, auc_pr_prev
                else:
                    auc_roc_prev, auc_pr_prev = auc_roc, auc_pr
                
                auc_roc_l.append(auc_roc)
                auc_pr_l.append(auc_pr)

            print('epoch:', start_epoch + epoch, 'loss:', np.mean(loss_l), 
                  'auc_roc:', np.mean(auc_roc_l), 'auc_pr:', np.mean(auc_pr_l))
            del loss_l, auc_roc_l, auc_pr_l

            # Evaluate validation set
            if epoch % 2:
                # Model checkpoint 
                self.save_model_weight('model', epoch)

                with torch.no_grad():
                    self.model.eval()
                    loss_l, auc_roc_l, auc_pr_l, recall_l, prec_l, acc_l, f1_l = [], [], [], [], [], [], []
                    for i, data_batch in tqdm(enumerate(self.val_dataloader, 0), unit="batch", total=len(self.val_dataloader)):
                        
                        predictions, label, loss = self.compute_pred_label_loss(data_batch)

                        (auc_roc, auc_pr, 
                        recall_sc, precision_sc, 
                        accuracy_sc, f1_sc) = self.compute_metrics(label, predictions)

                        if auc_roc == 0:
                            auc_roc, auc_pr = auc_roc_prev, auc_pr_prev
                        else:
                            auc_roc_prev, auc_pr_prev = auc_roc, auc_pr

                        loss_l.append(loss.detach().item())
                        auc_roc_l.append(auc_roc)
                        auc_pr_l.append(auc_pr)
                        recall_l.append(recall_sc)
                        prec_l.append(precision_sc)
                        acc_l.append(accuracy_sc)
                        f1_l.append(f1_sc)

                        del loss, auc_roc, auc_pr

                    print('epoch:', start_epoch + epoch, 'val_loss:', np.mean(loss_l), 'val_auc_roc:', np.mean(
                        auc_roc_l), 'val_auc_pr:', np.mean(auc_pr_l))
                    self.write_metrics(start_epoch + epoch, np.mean(loss_l), np.mean(auc_roc_l), 
                                       np.mean(auc_pr_l), np.mean(recall_sc), 
                                       np.mean(precision_sc), np.mean(accuracy_sc), 
                                       np.mean(f1_sc), i, 'val')

                    if epoch == 1:
                        old_loss = np.mean(loss_l)
                        
                    if np.mean(loss_l) < old_loss:
                        self.save_model_weight('model_best_loss', epoch)
                        old_loss = np.mean(loss_l)
                    if epoch > 200:
                        if epoch == 201:
                            old_pr = np.mean(auc_pr_l)
                        if np.mean(auc_pr_l) < old_pr:
                            self.save_model_weight('model_best_auc_pr', epoch)
                            old_pr = np.mean(auc_pr_l)

                    del loss_l, auc_roc_l, auc_pr_l

    def compute_pred_label_loss(self, data_batch, compute_loss=True, sigmoid=False):
        Tensor = torch.cuda.FloatTensor
        X, y = data_batch
        if torch.cuda.is_available():
            X = X.type(Tensor)
            y = y.type(Tensor)
        else: 
            X = torch.FloatTensor(X)
            y = torch.FloatTensor(y)

        X = Variable(X)
        y = Variable(y)

        current = self.model(X)

        if sigmoid:
            sigmoid = nn.Sigmoid()
            current = sigmoid(current)

        if torch.cuda.is_available():
            label = y.detach().cpu().numpy()
            predictions = current.detach().cpu().numpy()
        else:
            label = y.detach().numpy()
            predictions = current.detach().numpy()

        if compute_loss:
            loss = self.model.loss.bce_loss(y, current)
            return predictions, label, loss
        else:
            return predictions, label

    def compute_metrics(self, label, predictions, all=True):
        if len(np.unique(label)) > 1:
            auc_roc = roc_auc_score(
                y_true=label, y_score=predictions, average='micro')
            auc_pr = average_precision_score(
                y_true=label, y_score=predictions, average='micro')
        else: 
            auc_roc, auc_pr = 0, 0

        if all:
            label_arg = np.argmax(label, axis=1)
            predictions_arg = np.argmax(predictions, axis=1)

            recall_sc = recall_score(
                y_true=label_arg, y_pred=predictions_arg, average='micro')
            precision_sc = precision_score(
                y_true=label_arg, y_pred=predictions_arg, average='micro')
            accuracy_sc = accuracy_score(
                y_true=label_arg, y_pred=predictions_arg)
            f1_sc = f1_score(y_true=label_arg, y_pred=predictions_arg, average='micro')

            return auc_roc, auc_pr, recall_sc, precision_sc, accuracy_sc, f1_sc
        else:
            return auc_roc, auc_pr

    def feature_extract(self, device, epochs=100, best=0, start_directory='', learning_rate=0.001):
        self.load_weights(device, start_directory=start_directory, trained_model_dir=self.config.weights, best=best)

        ## freeze the layers
        # for param in self.model.parameters():
        #     param.requires_grad = False
            # ic(name, param.data)

        # Modify the last layer
        number_features = self.model.final1.dense_layer.in_features
        ic(number_features)
        self.model.final1 = Final(1, number_features, self.n_TFs_TL)
        self.model.to(device)

        # for name, param in self.model.named_parameters():
        #     if name.split('.')[0] == 'dense_block1':
        #         param.requires_grad = True
            # if param.requires_grad:
            # ic(name, param)

        for module, param in zip(self.model.modules(), self.model.parameters()):
            """However, we first need to pay close attention to the batch normalization layers in the network 
            architecture. These layers have specific mean and standard deviation values that were obtained 
            when the network was originally trained"""
            if isinstance(module, nn.BatchNorm1d):
                param.requires_grad = False

        print(self.model)
        self._set_optimizer(learning_rate=learning_rate)
        self.fit(epochs=epochs)

    def fine_tuning(self, device='cuda', epochs=100, start_epoch=0, start_directory='', weights=None, best=0, learning_rate=0.0002):
        self.load_weights(device, start_directory=start_directory, trained_model_dir=weights, best=best)

        ## unfreeze the layers
        for param in self.model.parameters():
            param.requires_grad = True

        # loop over the modules of the model and set the parameters of
# batch normalization modules as not trainable
        for module, param in zip(self.model.modules(), self.model.parameters()):
            """However, we first need to pay close attention to the batch normalization layers in the network 
            architecture. These layers have specific mean and standard deviation values that were obtained 
            when the network was originally trained"""
            if isinstance(module, nn.BatchNorm1d):
                param.requires_grad = False

        self.model.to(device)
        self._set_optimizer(learning_rate=learning_rate)
        self.fit(epochs, start_epoch)

    def write_metrics(self, epoch, loss, auc_roc, auc_pr, recall_sc, precision_sc, accuracy_sc, f1_sc, i, type_data):
        step = epoch if type_data == "val" else (epoch*len(self.train_dataloader)) + i
        
        # Tensorboard logs
        self.writer.add_scalar('Loss/' + type_data, loss, step)
        self.writer.add_scalar('Metric/auc_roc_' + type_data, auc_roc, step)
        self.writer.add_scalar('Metric/auc_pr_' + type_data, auc_pr, step)
        self.writer.add_scalar('Metric/recall_score_' + type_data, recall_sc, step)
        self.writer.add_scalar('Metric/precision_score_' + type_data, precision_sc, step)
        self.writer.add_scalar('Metric/accuracy_score_' + type_data, accuracy_sc, step)
        self.writer.add_scalar('Metric/f1_score_' + type_data, f1_sc, step)

    def save_model_weight(self, name, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path + name + '.pth')


    def load_weights(self, device, start_directory='../../', trained_model_dir=None, best=0):
        name = 'model_best_loss' if best==1 else 'model_best_auc_pr' if best==2 else 'model'
        trained_model_dir = self.model_path if trained_model_dir is None else trained_model_dir
        ic(trained_model_dir)
        checkpoint = torch.load(start_directory + trained_model_dir + name + '.pth', map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def predict_one(self, X, Y, item_id=0):
        Tensor = torch.cuda.FloatTensor
        with torch.no_grad():
            x_tmp = X[item_id:item_id+1]
            if torch.cuda.is_available():
                x_tmp = x_tmp.type(Tensor)
            
            x = Variable(x_tmp)
            y = Y[item_id]
            df_y = pd.DataFrame(y)

            self.model.eval()
            prediction = self.model(x)

            sigmoid = nn.Sigmoid()
            prediction = sigmoid(prediction)

            if torch.cuda.is_available():
                prediction = prediction.cpu()
            df_prediction = pd.DataFrame(prediction[0])

        return df_prediction, df_y 

    def predict_all(self, X, Y):
        Tensor = torch.cuda.FloatTensor
        with torch.no_grad():
            if torch.cuda.is_available():
                X = X.type(Tensor)

            x = Variable(X)
            df_y = pd.DataFrame(Y)

            self.model.eval()
            prediction = self.model(x)

            sigmoid = nn.Sigmoid()
            prediction = sigmoid(prediction)

            if torch.cuda.is_available():
                prediction = prediction.cpu()
            df_prediction = pd.DataFrame(prediction.numpy())

        return df_prediction, df_y 

    def predict_batch(self, data_loader, sigmoid=True):
        
        self.model.eval()
        with torch.no_grad():
            for i, data_batch in tqdm(enumerate(data_loader, 0), unit="batch", total=len(data_loader)):
                prediction, label = self.compute_pred_label_loss(data_batch, compute_loss=False, sigmoid=sigmoid)

                if i == 0:
                    predictions = prediction
                    labels = label
                else:
                    predictions = np.concatenate((predictions, prediction), axis=0)
                    labels = np.concatenate((labels, label), axis=0)
        
        df_y = pd.DataFrame(labels)
        df_prediction = pd.DataFrame(predictions)

        return df_prediction, df_y 

    def contruct_auc_scores_by_TF(self, df_y, df_pred, adata=None, only_TF=False):
        auc_roc_l, auc_pr_l = [], []
        for col in df_pred.columns:   
            y_true = df_y[col].values
            y_score = df_pred[col].values

            auc_roc, auc_pr = self.compute_metrics(y_true, y_score, all=False)

            auc_roc_l.append(auc_roc)
            auc_pr_l.append(auc_pr)

        df_total_score = pd.DataFrame(columns=['auc_roc', 'auc_pr'])
        df_total_score['auc_roc'] = auc_roc_l
        df_total_score['auc_pr'] = auc_pr_l
        df_total_score['count_regions'] = df_y[df_y == 1].count()
        if adata is not None: 
            df_total_score['TF'] = adata.obs.TF.values
            df_total_score['motif_model'] = adata.obs.motif_model.values
            if not only_TF:
                df_total_score['cell_line'] = adata.obs.cell_line.values
        return df_total_score