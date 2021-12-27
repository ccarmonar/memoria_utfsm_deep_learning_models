# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import gc

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import joblib
import os
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import datetime
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.max_open_warning': 0})

from featurize import SPARQLTreeFeaturizer
from early_stopping import EarlyStopping
from net import Autoencoder, NeoNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
CUDA = torch.cuda.is_available()

print(f"IS CUDA AVAILABLE: {CUDA}")


def _nn_path(base):
    return os.path.join(base, "nn_weights")


def _x_transform_path(base):
    return os.path.join(base, "x_transform")


def _y_transform_path(base):
    return os.path.join(base, "y_transform")


def _channels_path(base):
    return os.path.join(base, "channels")


def _n_path(base):
    return os.path.join(base, "n")


# General Methods
def scatter_image(y_pred, y_test, title, name, max_refference=300, figsize=None):
    plt.clf()
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.scatter(y_pred, y_test)
    plt.plot(range(max_refference))
    plt.xlabel("Prediction")
    plt.ylabel("Real latency")
    plt.savefig(name + '.png')
    plt.clf()


#     plt.show()


def plot_history(history):
    plt.clf()
    fig, axis = plt.subplots(1, 3)
    axis[0].plot(history['rmse_by_epoch'])
    axis[0].set_title('RMSE by epoch')
    axis[1].plot(history['mse_by_epoch'])
    axis[1].set_title("MSE by epoch")
    axis[2].plot(history['mae_by_epoch'])
    axis[2].set_title("MAE by epoch")
    fig.savefig("histories_mse_rmse_mae_valdataset" + '.png')
    plt.cla()


def scatter_plot_history(y_pred, y_true, y_predval, y_trueval, title_scatter, name, history, max_refference=300,
                         figsize=None, title_all="Scatter and history",start_history_from_epoch=1):
    plt.clf()

    fig, axis = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title_all, fontsize=16)
    colors = []
    markers = []
    colorsval = []
    markersval = []
    
    y_pred_l = [list(x) for x in y_pred]
    y_true_l = [list(x) for x in y_true]
    y_predval_l = [list(x) for x in y_predval]
    y_trueval_l = [list(x) for x in y_trueval]
    
    # Todo, cambiar para pintar diferentes marcadores.
    for pred, real in zip(y_pred, y_true):
        difference = real - pred
        abs_diff = np.abs(difference)
        p20 = real * 0.2
        p40 = real * 0.4
        if abs_diff < p20:
            colors.append("green")
            markers.append(".")
        elif abs_diff < p40:
            colors.append("blue")
            markers.append("x")
        else:
            colors.append("red")
            markers.append("d")

    for pred, real in zip(y_predval, y_trueval):
        difference = real - pred
        abs_diff = np.abs(difference)
        p20 = real * 0.2
        p40 = real * 0.4
        if abs_diff < p20:
            colorsval.append("green")
            markersval.append(".")
        elif abs_diff < p40:
            colorsval.append("blue")
            markersval.append("x")
        else:
            colorsval.append("red")
            markersval.append("d")

    axis[0, 0].set_title(f"{title_scatter} TrainSet")
    axis[0, 0].scatter(y_pred_l, y_true_l, c=colors, marker=".")
    axis[0, 0].plot(range(max_refference), 'g--')
    axis[0, 0].set_xlabel("Prediction")
    axis[0, 0].set_ylabel("Real latency")

    axis[1, 0].set_title(f"{title_scatter} ValidationSet")
    axis[1, 0].scatter(y_predval_l, y_trueval_l, c=colorsval, marker=".")
    axis[1, 0].plot(range(max_refference), 'g--')
    axis[1, 0].set_xlabel("Prediction")
    axis[1, 0].set_ylabel("Real latency")

    axis[0, 1].plot(history['rmse_by_epoch'][start_history_from_epoch:], label='train',marker=".")
    axis[0, 1].plot(history['rmse_val_by_epoch'][start_history_from_epoch:], label='validation',marker=".")
    axis[0, 1].set_title("RMSE by epoch")
    axis[0, 1].legend(loc="upper right")

    axis[1, 1].plot(history['mae_by_epoch'][start_history_from_epoch:], label='train',marker=".")
    axis[1, 1].plot(history['mae_val_by_epoch'][start_history_from_epoch:], label='validation',marker=".")
    axis[1, 1].set_title("MAE by epoch")
    axis[1, 1].legend(loc="upper right")

    fig.savefig(name + '.png')
    plt.cla()

def _inv_log1p(x):
    return np.exp(x) - 1

###################################################################
###################################################################

class NeoRegression:
    def __init__(self,
                 verbose=False,
                 aec=None,
                 epochs=100,
                 maxcardinality=1,
                 in_channels=None,
                 in_channels_neo_net=512,
                 tree_units=None,
                 tree_units_dense=None,
                 query_input_size=None,
                 query_hidden_inputs=None,
                 query_output=240,
                 early_stop_patience=10,
                 early_stop_initial_patience=10,
                 optimizer=None,
                 figimage_size=(10, 8),
                 tree_activation_tree=nn.LeakyReLU,
                 tree_activation_dense=nn.LeakyReLU,
                 ignore_first_aec_data=18,
                 start_history_from_epoch=2,
                 batch_size = 64
                 ):
        #print("NeoRegression, __init__ method active")
        if tree_units_dense is None:
            tree_units_dense = [32, 28]

        if tree_units is None:
            tree_units = [256, 128, 64]

        if query_hidden_inputs is None:
            query_hidden_inputs = [260, 300]

        if optimizer is None:
            optimizer = {'optimizer': 'Adam', 'args': {'lr': 0.00015}}
        if aec is None:
            aec = {'train_aec': False, 'aec_file': None, 'aec_epochs': 200}
        if aec['train_aec']:
            assert aec['aec_file'] is not None, "If train_aec is True, must define aec_file: path"
            assert (isinstance(aec['aec_epochs'], int) and aec[
                'aec_epochs'] > 0), "If train_aec is True, must define aec_epochs: int"
        self.net = None
        self.verbose = verbose
        self.train_aec = aec['train_aec']
        self.aec_file = aec['aec_file']
        self.aec_epochs = aec['aec_epochs']
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.early_stop_initial_patience = early_stop_initial_patience
        # This is the output units for encoder of autoencoder model.
        self.in_channels_neo_net = in_channels_neo_net
        self.ignore_first_aec_data = ignore_first_aec_data
        self.query_input_size = query_input_size
        self.query_hidden_inputs = query_hidden_inputs
        self.query_output = query_output
        self.best_model = None
        self.optimizer = optimizer
        print(f"Model optimizer: {self.optimizer['optimizer']} lr: {self.optimizer['args']['lr']}")
        log_transformer = preprocessing.FunctionTransformer(
            np.log1p, _inv_log1p,
            validate=True)
        scale_transformer = preprocessing.MinMaxScaler()

        self.pipeline = Pipeline([("log", log_transformer), ("scale", scale_transformer)])

        self.tree_transform = SPARQLTreeFeaturizer()
        self.in_channels = in_channels
        self.n = 0
        self.aec_net = None
        self.figimage_size = figimage_size

        # configs of tree model
        self.tree_units = tree_units
        self.tree_units_dense = tree_units_dense
        # configure the activation function of tree convolution layers(see model)
        self.tree_activation_tree = tree_activation_tree
        # configure the activation function of tree convolution dense layer(see model)
        self.tree_activation_dense = tree_activation_dense
        self.start_history_from_epoch=start_history_from_epoch
        
        self.history = {
            'rmse_by_epoch': [],
            'mse_by_epoch': [],
            'mae_by_epoch': [],
            'rmse_val_by_epoch': [],
            'mse_val_by_epoch': [],
            'mae_val_by_epoch': []
        }
        self.maxcardinality = maxcardinality
        self.batch_size = batch_size
    def log(self, *args):
        print("NeoRegression, log method active")
        if self.verbose:
            print(*args)

    def num_items_trained_on(self):
        print("NeoRegression, num_items_trained_on method active")
        return self.n

    def get_pred(self):
        #print("NeoRegression, get_pred method active")
        return self.tree_transform.get_pred_index()

    def load(self, path, best_model_path):
        #print("NeoRegression, load method active")
        with open(_n_path(path), "rb") as f:
            self.n = joblib.load(f)
        with open(_channels_path(path), "rb") as f:
            self.in_channels = joblib.load(f)

        self.net = NeoNet(self.aec_net, self.in_channels_neo_net, self.query_input_size,
                            self.query_hidden_inputs, self.query_output)

        if best_model_path is not None:
            self.net.load_state_dict(torch.load(best_model_path))
        else:
            self.net.load_state_dict(torch.load(_nn_path(path)))
        self.net.cuda()
        self.net.eval()

        with open(_y_transform_path(path), "rb") as f:
            self.pipeline = joblib.load(f)
        with open(_x_transform_path(path), "rb") as f:
            self.tree_transform = joblib.load(f)

    def fix_tree(self, tree):
        #print("NeoRegression, fix_tree method active")
        """
        Trees in data must include in first position join type follow by predicates of childs. We check and fix this.
        """

        try:
            if len(tree) == 1:
                assert (isinstance(tree[0], str))
                return tree
            else:
                assert (len(tree) == 3)
                assert (isinstance(tree[0], str))
                preds = []
                if len(tree[0].split("ᶲ")) == 1:

                    tree_left = self.fix_tree(tree[1])
                    preds.extend(tree_left[0].split("ᶲ")[1:])

                    tree_right = self.fix_tree(tree[2])
                    preds.extend(tree_right[0].split("ᶲ")[1:])
                    preds = list(set(preds))
                    tree[0] = tree[0] + "ᶲ" + "ᶲ".join(preds)
                    return tree
                else:
                    return tree

        except Exception as ex:
            
            return tree

    def save(self, path):
        print("NeoRegression, save method active")
        # try to create a directory here
        os.makedirs(path, exist_ok=True)

        torch.save(self.net.state_dict(), _nn_path(path))
        with open(_y_transform_path(path), "wb") as f:
            joblib.dump(self.pipeline, f)
        with open(_x_transform_path(path), "wb") as f:
            joblib.dump(self.tree_transform, f)
        with open(_channels_path(path), "wb") as f:
            joblib.dump(self.in_channels, f)
        with open(_n_path(path), "wb") as f:
            joblib.dump(self.n, f)

    def fit_transform_tree_data(self, ds_train, ds_val, ds_test,ds_rl):
        #print("NeoRegression, fit_transform_tree_data method active")
        ds_train = self.json_loads_trees_ds(ds_train)
        ds_val = self.json_loads_trees_ds(ds_val)
        ds_test = self.json_loads_trees_ds(ds_test)
        ds_rl = self.json_loads_trees_ds(ds_rl)
        data = []
        data.extend(ds_train)
        data.extend(ds_val)
        data.extend(ds_test)
        data.extend(ds_rl)

        self.tree_transform.fit(data)
        
    def fit_transform_tree_data_no_ds_rl(self, ds_train, ds_val, ds_test):
        #print("NeoRegression, fit_transform_tree_data method active")
        ds_train = self.json_loads_trees_ds(ds_train)
        ds_val = self.json_loads_trees_ds(ds_val)
        ds_test = self.json_loads_trees_ds(ds_test)
        data = []
        data.extend(ds_train)
        data.extend(ds_val)
        data.extend(ds_test)

        self.tree_transform.fit(data)

    def transform_trees(self, data):
        print("NeoRegression, transform_trees method active")
        return self.tree_transform.transform(data)

    def load_aec(self):
        print("NeoRegression, load_aec method active")
        self.log("Loading pretrained Autoencoder", "...")
        self.aec_net = Autoencoder(self.in_channels)
        self.aec_net.load_state_dict(torch.load(self.aec_file))
        self.aec_net.cuda()
        self.aec_net.eval()
        return self.aec_net

    def fit(self, X, X_query, y, X_val, X_val_query, y_val):
        #print("NeoRegression, fit method active")
        if isinstance(y, list):
            y = np.array(y)
    
        X,X_query, y = self.json_loads(X, X_query, y)

        
        X = [self.fix_tree(x) for x in X] 
        print("X_train loaded")
        
        X_val,X_val_query, y_val = self.json_loads(X_val, X_val_query, y_val)
        X_val = [self.fix_tree(x) for x in X_val]
        print("X_val loaded")

        self.n = len(X)
        max_y = np.max(y)

        # Fit target transformer
        self.pipeline.fit_transform(y.reshape(-1, 1))
   
        #print("Transforming Trees: se pasan a una tupla donde cada número representa lo que se obtiene en reg.fit_transform_tree_data(ds_train, ds_val, ds_test)")
        X = self.tree_transform.transform(X)
        X_val = self.tree_transform.transform(X_val)
        # determine the initial number of channels
        io_dim = len(self.get_pred())
        print("determine the initial number of channels, io_dim", io_dim)
        pairs = list(zip(list(zip(X, X_query)), y))
        pairs_val = list(zip(list(zip(X_val, X_val_query)), y_val))
        print("self.batch_size",self.batch_size)
        if self.maxcardinality == 0:
            # Case for no cardinalities encoded data
            dataset = DataLoader(pairs, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=self.collate)
            dataset_val = DataLoader(pairs_val, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=self.collate)
        else:
            dataset = DataLoader(pairs, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=self.collate_with_card)
            dataset_val = DataLoader(pairs_val, batch_size=self.batch_size, num_workers=0, shuffle=True, collate_fn=self.collate_with_card)

        self.query_input_size = len(X_query[0])
        if self.maxcardinality != 0:
            # Case when cardinalities are added. We need to extend queries features to queries features+ len(pred2index) - 1
            self.query_input_size = self.query_input_size+io_dim-1
        print("Initial input channels of tree model:", self.in_channels)
        print("io_dim", io_dim)
        print("self.query_input_size", self.query_input_size)
        print("self.query_hidden_inputs", self.query_hidden_inputs)
        print("self.query_output", self.query_output)
        print("self.tree_units", self.tree_units)
        print("self.tree_units_dense", self.tree_units_dense)
        print("self.tree_activation_tree", self.tree_activation_tree)
        print("self.tree_activation_dense",self.tree_activation_dense)
        self.net = NeoNet(
            io_dim,
            self.query_input_size,
            self.query_hidden_inputs,
            self.query_output,
            tree_units=self.tree_units,
            tree_units_dense=self.tree_units_dense,
            activation_tree=self.tree_activation_tree,
            activation_dense=self.tree_activation_dense
        )
        if CUDA:
            self.net = self.net.cuda()

        # initialize the early_stopping object
        early_stopping = EarlyStopping(initial_patience=self.early_stop_initial_patience,
                                       patience=self.early_stop_patience, verbose=True)
        
        if self.optimizer["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.net.parameters(), **self.optimizer["args"])
        elif self.optimizer["optimizer"] == "Adagrad":
            optimizer = torch.optim.Adagrad(self.net.parameters(), **self.optimizer["args"])
        else:
            optimizer = torch.optim.SGD(self.net.parameters(), **self.optimizer["args"])
        print("OPTIMIZADOr A UTILIZAR: ", optimizer)
        
        loss_fn = torch.nn.MSELoss()

        losses = []
        
        # Fot ID of images
        import random
        id_label = "".join([str(a) for a in random.sample(range(20), 5)])
        self.id_label = id_label
        if not os.path.exists("./" + id_label + "/"):
            os.makedirs("./" + id_label + "/")

        assert np.mean(y_val) > 5, "y_val must be in real scale"
        print("Max epochs to run:", self.epochs)
        for epoch in range(self.epochs):
            self.net.train()
            loss_accum = 0
            results_train = []
            #### A TRAVES DEL DATALOADER EN EL FOR SE UTILIZAN LOS METODOS NeoRegression, collate_with_card method active
            ####  NeoRegression, get_pred method active LOS CUALES SON PREPROCESSAMIENTOS PREVIOS
            for (x, y_train) in dataset:
                y_train_scaled = torch.tensor(self.pipeline.transform(y_train.reshape(-1, 1)).astype(np.float32))
                if CUDA:
                    y_train_scaled = y_train_scaled.cuda()
                
                tipos = []
            
                
                #print("len(x)",len(x))
                #for i in x:
                #    print("type i[0]",type(i[0]),"len i[0]",len(i[0]))
                #    print("type i[1]",type(i[1]),"len i[1]",len(i[1]))
                
                y_pred = self.net(x)
                loss = loss_fn(y_pred, y_train_scaled)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lost_item = loss.item()
                loss_accum += lost_item
                results_train.extend(
                    list(zip(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()), y_train)))
                
            ###########quede mas o menos por aqui
            loss_accum /= len(dataset)
            losses.append(loss_accum)

            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_accum / len(dataset)))
            
            # Prediction in subsample of train
            torch.cuda.empty_cache()

            y_pred_train, y_real_train = zip(*results_train)
            msetrain = mean_squared_error(y_real_train, y_pred_train)
            maetrain = mean_absolute_error(y_real_train, y_pred_train)
            rmsetrain = np.sqrt(msetrain)
            self.history['mse_by_epoch'].append(msetrain)
            self.history['rmse_by_epoch'].append(rmsetrain)
            self.history['mae_by_epoch'].append(maetrain)

            # Testing the model

            results_val = self.predict(dataset_val)
            y_pred_val, y_real_val = zip(*results_val)
            mseval = mean_squared_error(y_real_val, y_pred_val)
            maeval = mean_absolute_error(y_real_val, y_pred_val)
            rmseval = np.sqrt(mseval)
            self.history['mse_val_by_epoch'].append(mseval)
            self.history['rmse_val_by_epoch'].append(rmseval)
            self.history['mae_val_by_epoch'].append(maeval)
            #             print(f"RMSE in TRAIN: {rmsetrain} : RMSE in VAL: {rmseval}")
            #print("***************************")
            print('==> Epoch {},\tTRAIN_LOSS: {}\t_TRAIN_RMSE: {},\tVAL_LOSS: {},\tVAL_RMSE: {}'.format(
                epoch, msetrain, rmsetrain, mseval, rmseval))
            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            best_model = early_stopping(np.average(self.history['rmse_val_by_epoch'][-self.early_stop_patience:]),
                                        self.net)
            if best_model is not None:
                self.best_model = best_model
            if early_stopping.early_stop:
                print("Early stopping the training.")
                break

            if epoch % 4 == 0:
                self.scatter_plot_history(
                    y_pred_train,
                    y_real_train,
                    y_pred_val,
                    y_real_val,
                    "Scatter real latency vs prediction on: ",
                    "./" + id_label + "/" + "neo_with_aec_scatter_train_val_epoch_" + "{:03d}".format(epoch),
                    self.history,
                    max_refference=int(max_y + 10),
                    figsize=self.figimage_size,
                    title_all=f"Scatter and history, RMSE Train: {rmsetrain}, RMSE VAL: {rmseval}, Epoch: {epoch}",
                    start_history_from_epoch=self.start_history_from_epoch
                )
            gc.collect()

    def predict(self, val_loader):
        #print("NeoRegression, predict method active")
        results = []
        self.net.eval()
        with torch.no_grad():
            for (x, y_val) in val_loader:
                y_pred = self.net(x)
                results.extend(list(zip(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()), y_val)))
        return results

    def predict_raw_data(self, trees, queries):
        #print("NeoRegression, predict method active")
        results = []
        pares = list(zip(trees, queries))
        dataloader = DataLoader(pares, batch_size=128, shuffle=False, collate_fn=self.collate2)
        self.net.eval()
        with torch.no_grad():
            for x in dataloader:
                y_pred = self.net(x)
                results.extend(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()))
        return results

    def predict_best(self, val_loader):
        #print("NeoRegression, predict_best method active")
        results = []
        self.best_model.eval()
        with torch.no_grad():
            for (x, y_val) in val_loader:
                y_pred = self.best_model(x)
                results.extend(list(zip(self.pipeline.inverse_transform(y_pred.cpu().detach().numpy()), y_val)))
        return results

    def index2sparse(self, tree, sizeindexes):
        #print("NeoRegression, index2sparse method active")
        resp = []
        for el in tree:
            if type(el[0]) == tuple:
                resp.append(self.index2sparse(el, sizeindexes))
            else:
                a = np.array(el)
                b = np.zeros((a.size, sizeindexes))
                b[np.arange(a.size), a] = 1
                onehot = np.sum(b, axis=0, keepdims=True)[0]
                resp.append(onehot)
        return tuple(resp)
    
    def index2sparse2(self, tree, sizeindexes):
        #print("NeoRegression, index2sparse2 method active")
        resp = []
        for el in tree:
            if type(el[0]) == tuple:
                resp.append(self.index2sparse2(el, sizeindexes))
            else:
                a = np.array(el)
                b = np.zeros((a.size, sizeindexes))
                b[np.arange(a.size), a] = 1
                onehot = np.sum(b, axis=0, keepdims=True)[0]
                resp.append(onehot)
        return tuple(resp)

    def collate_with_card(self, x):
        #print("NeoRegression, collate_with_card method active")
        """Preprocess inputs values, transform index2vec values, them predict aec.encoder to dimensionality reduction"""
        trees = []
        targets = []
        sizeindexes = len(self.get_pred())
        for tree, target in x:
            b = np.zeros((sizeindexes))
            try:
                for key in tree[1][-1].keys():
                    b[key] = tree[1][-1][key]
            except:
                print("Error en cardinalidades",str(tree[1][-1]))
            trees.append(tuple([self.index2sparse(tree[0], sizeindexes), np.concatenate([tree[1][:-1], b]).tolist()]))
            targets.append(target)

        targets = torch.tensor(targets)
        return trees, targets
    
    def collate(self, x):
        #print("NeoRegression, collate method active")
        """Preprocess inputs values, transform index2vec values, them predict aec.encoder to dimensionality reduction"""
        trees = []
        targets = []
        sizeindexes = len(self.get_pred())
        for tree, target in x:
            trees.append(tuple([self.index2sparse(tree[0], sizeindexes), tree[1]]))
            targets.append(target)

        targets = torch.tensor(targets)
        return trees, targets

    def collate2(self, x):
        #print("NeoRegression, collate2 method active")
        """Only collocate x_data"""
        trees = []
        sizeindexes = len(self.get_pred())
        for tree in x:
            trees.append(tuple([self.index2sparse(tree[0], sizeindexes), tree[1]]))
        return trees

    def json_loads(self, X, X_query, Y):
        #print("NeoRegression, json_loads method active")
        """read string with json data as json object"""
        respX = []
        respX_query = []
        respY = []
        for x, y in list(zip(list(zip(X, X_query)), Y)):
            try:
                x_tree, x_query = x
                x_tree = json.loads(x_tree)
                respX.append(x_tree)
                respX_query.append(x_query)
                respY.append(y)
            except:
                print("Error in data ignored!", x)
        return respX, respX_query, np.array(respY).reshape(-1, 1)
    
    
    

    def json_loads_trees_ds(self, ds):
        #print("NeoRegression, json_loads_trees_ds method active")
        """read string with json data as json object from Dataframe, ignore bad jsons trees"""
        respX = []
        for index in range(ds.shape[0]):
            row = ds.iloc[index]
            try:
                data = json.loads(row['trees'])
                respX.append(data)
            except:
                print("Error in data ignored!", row['trees'])
        return respX

    def scatter_image(self, y_pred, y_test, title, name, max_refference=300, figsize=None):
        #print("NeoRegression, scatter_image method active")
        scatter_image(y_pred, y_test, title, name, max_refference=max_refference, figsize=figsize)

    def plot_history(self, history):
        #print("NeoRegression, plot_history method active")
        plot_history(history)

    def scatter_plot_history(
        self, 
        y_pred, 
        y_true, 
        y_predval, 
        y_trueval, 
        title_scatter, 
        name,
        history,
        max_refference=300,
        figsize=None,
        title_all="Scatter and history",
        start_history_from_epoch=1):
        
        #print("NeoRegression, scatter_plot_history method active")
        
        scatter_plot_history(y_pred, y_true, y_predval, y_trueval, title_scatter, name, history,
                             max_refference=max_refference,
                             figsize=figsize, title_all=title_all,
                            start_history_from_epoch=start_history_from_epoch)

    def get_bestmodel(self):
        print("NeoRegression, get_bestmodel method active")
        return self.best_model

    def get_model(self):
        print("NeoRegression, get_model method active")
        return self.net
