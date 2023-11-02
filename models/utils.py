import datetime
import os, logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


class Sampler():  
    def __init__(self, dataloaders):
        self.wk_num = len(dataloaders)
        self.dataloaders = dataloaders
        self.iterators = self.get_iterators()

    def get_iterators(self):
        iterators = []
        for i in range(self.wk_num):
            # wk = self.wk_num[i]
            iterators.append(iter(self.dataloaders[i]))
        return iterators

    def get_sample(self):
        samples = {}
        for i in range(self.wk_num):
            # wk = self.wk_num[i]            
            samples[i] = next(self.iterators[i])
        return samples


def MAML_dataset(entire_X_tr, entire_y_tr, entire_X_te, entire_y_te, scaler_X, scaler_y, wk, batch_size=1):
    DL_tr = []
    DL_te = []  
    test_X_te = pd.DataFrame()
    test_y_te = pd.DataFrame()  
    for i in range(len(wk)):
        X_tr_ = entire_X_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:] #train data set for each workload is 16000
        y_tr_ = entire_y_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:]
        X_te_ = entire_X_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:]   #test data set for each workload is 4000
        y_te_ = entire_y_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:] 
       
        s_X_tr = torch.Tensor(scaler_X.transform(X_tr_)).cuda()
        s_X_te = torch.Tensor(scaler_X.transform(X_te_)).cuda()
        s_y_tr = torch.Tensor(scaler_y.transform(y_tr_)).cuda()
        s_y_te = torch.Tensor(scaler_y.transform(y_te_)).cuda()

        # s_X_tr = torch.Tensor(X_tr_.values).cuda()
        # s_X_te = torch.Tensor(X_te_.values).cuda()
        # s_y_tr = torch.Tensor(y_tr_.values).cuda()
        # s_y_te = torch.Tensor(y_te_.values).cuda()
        test_X_te = pd.concat((test_X_te, X_te_))
        test_y_te = pd.concat((test_y_te, y_te_))
        

        dataset_tr = TensorDataset(s_X_tr, s_y_tr)
        dataset_te = TensorDataset(s_X_te, s_y_te)
        dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        dataloader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
        DL_tr.append(dataloader_tr)
        DL_te.append(dataloader_te)

    s_test_X_te = torch.Tensor(scaler_X.transform(test_X_te)).cuda()
    s_test_y_te = torch.Tensor(scaler_y.transform(test_y_te)).cuda()

    return DL_tr, DL_te, s_test_X_te, s_test_y_te


def pretrain_dataset(entire_X_tr, entire_y_tr, entire_X_te, entire_y_te, scaler_X, scaler_y, wk, batch_size=1):   # wk : using workload ex): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  
    ## MODIFIED 220422
    selected_X_tr = pd.DataFrame()
    selected_y_tr = pd.DataFrame()
    selected_X_te = pd.DataFrame()
    selected_y_te = pd.DataFrame()

    test_X_te = pd.DataFrame()
    test_y_te = pd.DataFrame()  
    for i in range(len(wk)):
        X_tr_ = entire_X_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:] #train data set for each workload is 16000
        y_tr_ = entire_y_tr.iloc[(16000*wk[i]):16000*(wk[i]+1),:]
        X_te_ = entire_X_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:]   #test data set for each workload is 4000
        y_te_ = entire_y_te.iloc[(4000*wk[i]):4000*(wk[i]+1),:] 
        
        selected_X_tr = pd.concat((selected_X_tr, X_tr_))
        selected_y_tr = pd.concat((selected_y_tr, y_tr_))
        selected_X_te = pd.concat((selected_X_te, X_te_))
        selected_y_te = pd.concat((selected_y_te, y_te_))

        test_X_te = pd.concat((test_X_te, X_te_))
        test_y_te = pd.concat((test_y_te, y_te_))
        
    s_X_tr = torch.Tensor(scaler_X.transform(selected_X_tr)).cuda()
    s_X_te = torch.Tensor(scaler_X.transform(selected_X_te)).cuda()
    s_y_tr = torch.Tensor(scaler_y.transform(selected_y_tr)).cuda()
    s_y_te = torch.Tensor(scaler_y.transform(selected_y_te)).cuda()      

        # dataset_tr = TensorDataset(s_X_tr, s_y_tr)
        # dataset_te = TensorDataset(s_X_te, s_y_te)
        # DL_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        # DL_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)
    
    ## MODIFIED 220422
    dataset_tr = TensorDataset(s_X_tr, s_y_tr)
    dataset_te = TensorDataset(s_X_te, s_y_te)
    DL_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
    DL_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=True)

    s_test_X_te = torch.Tensor(scaler_X.transform(test_X_te)).cuda()
    s_test_y_te = torch.Tensor(scaler_y.transform(test_y_te)).cuda()

    return DL_tr, DL_te, s_test_X_te, s_test_y_te

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
