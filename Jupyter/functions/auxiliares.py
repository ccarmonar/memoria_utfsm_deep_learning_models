import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import math
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler#para escalar caracteristicas
from sklearn.model_selection import train_test_split # separar mas facil la data de train y test
import pandas as pd
from sklearn.utils import shuffle
from ast import literal_eval
import time
import sys
import os


def PaddingSameSize(matrix,padd_type='constant',constant_value=0):
    length = np.array([len(matrix[i]) for i in range(len(matrix))])
    width = length.max()
    return_list=[]
    for i in range(len(matrix)):
        if len(matrix[i]) != width:
            if padd_type == 'constant':
                padd = np.pad(matrix[i], (0,width-len(matrix[i])), 'constant',constant_values = 0)
            else:
                padd = np.pad(matrix[i], (0,width-len(matrix[i])), padd_type)
        else:
            padd = matrix[i]
        return_list.append(padd)
    return_list = np.array(return_list)
    return return_list


def MatrixFormat_To_Vector(df_matrix_format):
    df_matrix_format = df_matrix_format.apply(lambda x: np.asarray(literal_eval(x)).astype(np.float32)) 
    max_x,max_y = {"index":0,"value":0},{"index":0,"value":0}
    vector_list = []
    vector_list_pad = []
    max_len = 0
    for i, matrix in df_matrix_format.items():
    #    if max_x['value'] < matrix.shape[0]:
    #        max_x['value'] = matrix.shape[0]
    #        max_x['index'] = i
    #    if max_y['value'] < matrix.shape[1]:
    #        max_y['value'] = matrix.shape[1]
    #        max_y['index'] = i
        vector = np.reshape(matrix, -1)
        vector_list.append(vector)

    #for v in vector_list:
    #    if max_len < len(v):
    #        max_len = len(v)
    #print("max len", max_len)
    vector_list_pad = PaddingSameSize(vector_list, 'constant')      
    
    #for i in vector_list_pad:
    #    print(i.shape)
    #    print(i)
    
    return vector_list_pad
    
    
    
def RemoveOversizedMatrix(df_clean,max_r=20,max_c=40):
    arr_shape_r = []
    arr_shape_c = []
    print("Total df_clean: ",len(df_clean))
    eliminados = []
    for i, v in df_clean['matrix_format'].iteritems():
        r,c = v.shape
        if r > 20 or c > 40:
            eliminados.append(i)
    print("Eliminados por Oversized:", len(eliminados))

    new_df_clean = df_clean.drop(eliminados)

    for i, v in new_df_clean['matrix_format'].iteritems():
        r,c = v.shape
        arr_shape_r.append(r)
        arr_shape_c.append(c)
        
    max_new_r = max(arr_shape_r)
    max_new_c = max(arr_shape_c)
    min_new_r = min(arr_shape_r)
    min_new_c = min(arr_shape_c)
    mean_new_r = np.ceil(np.mean(arr_shape_r))
    mean_new_c = np.ceil(np.mean(arr_shape_c))

    return new_df_clean,max_new_r,max_new_c,min_new_r,min_new_c,mean_new_r,mean_new_c
    
    
    
def StandardSize_Padding(matrix,num_standard_rows,num_standard_columns):
    padd_columns = []
    for i in range(len(matrix)):
        if len(matrix[i]) < num_standard_columns:
            padd = np.pad(matrix[i], (0,num_standard_columns-len(matrix[i])), 'mean')
            padd_columns.append(padd)
        else:
            padd = matrix[i]
            padd_columns.append(padd)
    padd_rows = []
    tp_matrix = np.asarray(padd_columns, dtype=np.float32).T
    for i in range(len(tp_matrix)):
        if len(tp_matrix[i]) < num_standard_rows:
            padd = np.pad(tp_matrix[i], (0,num_standard_rows-len(tp_matrix[i])), 'mean')
            padd_rows.append(padd)
        else:
            padd = tp_matrix[i]
            padd_rows.append(padd)
    return np.asarray(padd_rows, dtype=np.float32).T
    
    
def CNN_Features_Format(numpy_array):
    ### Pasarlos a 1 "canal"
    canal = []
    #print("shape X_numpy_train: ",numpy_array.shape)
    #print("type X_numpy_train: ",type(numpy_array), numpy_array.dtype)
    for i in numpy_array:
        for j in i:
            k = np.array([j])
            canal.append(k)
    canal = np.array(canal)
    torch_canal = torch.from_numpy(canal)
    return torch_canal
    
    
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
    
def batch_format(X_train, y_train, n=1):
    batches = {"features":[],"labels":[],}
    for x in batch(X_train, n):
        batches["features"].append(x)
    for y in batch(y_train, n):
        batches["labels"].append(y)
    return batches