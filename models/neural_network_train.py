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
        # epoch_loss = 0 ####
        trigger_times = 0
        breaker = False ####
        for e in range(1, self.num_epochs+1):    # epoch
            sampler_tr = Sampler(dataloaders=self.train_dataloaders)
            #################################################################
            sampler_val = Sampler(dataloaders=self.val_dataloaders)
            #################################################################    
            for iter in range(1, len(self.train_dataloaders[0])+1):    # iteration       
                total_meta_loss_tr = 0
   
                self.meta_optimizer.zero_grad()
                sample_tr = sampler_tr.get_sample()
                sample_val = sampler_val.get_sample()
                meta_loss_sum = 0
                wk_loss =[]                     

                # inner loop
                for num_wk in range(self.num_meta_tasks):
                    self.sample_tr = sample_tr[num_wk]
                    self.sample_val = sample_val[num_wk]  ###                  
                    _, meta_loss = self.inner_loop(iter)
                    wk_loss.append(meta_loss)

                    meta_loss_sum += meta_loss   # i: task     

                if self.meta_mean ==True :
                    meta_loss_sum /= self.num_meta_tasks    ####

                total_meta_loss_tr += meta_loss_sum.item()
                meta_loss_tr = total_meta_loss_tr/len(self.train_dataloaders)
                
                # compute meta gradient of loss with respect to maml weights
                meta_loss_sum.backward()
                self.meta_optimizer.step()

            loss_te, te_outputs, r2_res = self.validate(self.model)  ###

            self.meta_losses_tr +=[meta_loss_tr]
            self.meta_losses_te += [loss_te]       
            self.r2_score += [r2_res]      

            time_ = datetime.datetime.today().strftime('%y%m%d/%H:%M:%S')
            print(f"{time_}[{e:02d}/{self.num_epochs}] meta_loss: {meta_loss_sum:.4f} loss_te: {loss_te:.4f}, r2_res = {r2_res:.4f}")
    
    def inner_loop(self, iter):     # i: task , iteration : iteration
        pass