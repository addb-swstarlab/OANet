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
        
    s_test_X_te = torch.Tensor(scaler_x.transform(test_X_te)).cuda()
    s_test_y_te = torch.Tensor(scaler_y.transform(test_y_te)).cuda()

    return DL_tr, DL_te, s_test_X_te, s_test_y_te

class Sampler():  
    def __init__(self, dataloaders):
        self.wk_num = len(dataloaders)
        self.dataloaders = dataloaders
        self.iterators = self.get_iterators()

    def get_iterators(self):
        iterators = []
        for i in range(self.wk_num):
            iterators.append(iter(self.dataloaders[i]))
        return iterators
    
    def get_sample(self):
        samples = {}
        for i in range(self.wk_num):
            samples[i] = next(self.iterators[i])
        return samples
    
def get_filename(PATH, head, tail):
    i = 0
    today = datetime.datetime.now()
    today = today.strftime('%Y%m%d')
    if not os.path.exists(os.path.join(PATH, today)):
        os.mkdir(os.path.join(PATH, today))
    name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    while os.path.exists(os.path.join(PATH, name)):
        i += 1
        name = today+'/'+head+'-'+today+'-'+'%02d'%i+tail
    return name


def get_logger(log_path='./logs'):

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = logging.getLogger()
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s[%(levelname)s] %(filename)s:%(lineno)s  %(message)s', date_format)
    name = get_filename(log_path, 'log', '.log')
    
    fileHandler = logging.FileHandler(os.path.join(log_path, name))
    streamHandler = logging.StreamHandler()
    
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    logger.setLevel(logging.INFO)
    logger.info('Writing logs at {}'.format(os.path.join(log_path, name)))
    return logger, os.path.join(log_path, name)
