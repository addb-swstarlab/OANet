import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score
from models.utils import get_filename, Sampler

class MAML_trainer():
    def __init__(self, model, train_dataloaders, val_dataloaders, test_dataloaders, num_epochs, inner_lr, meta_lr, inner_steps=1, meta_mean=False, dot=True, lamb=0.6):

        self.patience = 10  # early stopping 
        self.meta_mean = meta_mean

        self.model = model        
        self.train_dataloaders = train_dataloaders
        self.val_dataloaders = val_dataloaders
        self.test_dataloaders = test_dataloaders

        self.num_epochs = num_epochs
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.dot = dot
        self.lamb = lamb

        self.num_meta_tasks = len(self.train_dataloaders)  
        self.criterion = nn.MSELoss()
        self.weights = list(self.model.parameters()) 
        self.meta_optimizer = torch.optim.Adam(self.weights, self.meta_lr)        
        self.inner_steps = inner_steps      
        
        # metrics
        self.meta_losses_tr = []
        self.meta_losses_te = []
        self.r2_score = []
        self.best_loss = np.inf

    def main_loop(self):
        pass
    
    def inner_loop(self, iter):     # i: task , iteration : iteration
        pass