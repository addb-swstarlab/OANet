import datetime
import os, logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def MAML_dataset(X_tr, Y_tr, X_te, Y_te, scaler_x, scaler_y, wk, batch_size=1):
    DL_tr = []
    DL_te = []  
    test_X_te = pd.DataFrame()
    test_y_te = pd.DataFrame()
    for i in range(len(wk)):
        X_tr_, X_te_ = X_tr[i], X_te[i]        
        y_tr_, y_te_ = Y_tr[i], Y_te[i]
       
        norm_X_tr = torch.Tensor(scaler_x.transform(X_tr_)).cuda()
        norm_X_te = torch.Tensor(scaler_x.transform(X_te_)).cuda()
        norm_y_tr = torch.Tensor(scaler_y.transform(y_tr_)).cuda()
        norm_y_te = torch.Tensor(scaler_y.transform(y_te_)).cuda()

        test_X_te = pd.concat((test_X_te, X_te_))
        test_y_te = pd.concat((test_y_te, y_te_))        

        dataset_tr = TensorDataset(norm_X_tr, norm_y_tr)
        dataset_te = TensorDataset(norm_X_te, norm_y_te)
        dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        dataloader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
        DL_tr.append(dataloader_tr)
        DL_te.append(dataloader_te)

class Sampler():  
    pass

    
def get_filename(PATH, head, tail):
    pass


def get_logger(log_path='./logs'):
    pass
