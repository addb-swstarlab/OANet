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
    
                # early stopping
            if loss_te > self.best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print('Early stopping! \n training step finish')
                    breaker = True
            else:
                self.best_loss = loss_te
                self.best_ouputs = te_outputs
                self.best_r2_score = r2_res
                self.best_model = self.model
                self.best_epoch_num = e
                print('Trigger Times: 0')
                trigger_times = 0  

            print(f"{time_}[{e:02d}/{self.num_epochs}] meta_loss: {meta_loss_sum:.4f} loss_te: {loss_te:.4f}, r2_res = {r2_res:.4f} / trrigger times : {trigger_times}")  

            if breaker == True:
                break
            ###############################################################


        self.name = get_filename('model_save', 'MAML_train', '.pt')
        torch.save(self.best_model.state_dict(), os.path.join('model_save', self.name))  # save only state_dict of model
        print(f'Saved model state dict! name : {self.name}')
    
    def inner_loop(self, iter):     # i: task , iteration : iteration
        # copy inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]       
          
        # training on data sampled from each task
        X, y = self.sample_tr[0], self.sample_tr[1]
        ##########################################################
        X_val, y_val = self.sample_val[0], self.sample_val[1]
        ###########################################################
        inner_loss = self.criterion(self.model.parameterised(X, temp_weights), y)
        grad = torch.autograd.grad(inner_loss, temp_weights)
        temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        temp_pred = self.model.parameterised(X_val, temp_weights)
        
        # calculate loss for update maml weight (with update inner loop weight)
        if self.dot:
            d = torch.bmm(self.model.p_res_x, self.model.p_res_x.transpose(1, 2))
            dot_loss = F.mse_loss(d, torch.eye(d.size(1)).repeat(X.shape[0], 1, 1).cuda())
            meta_loss = (1-self.lamb)*self.criterion(temp_pred, y) + self.lamb*dot_loss
        else:
            meta_loss = self.self.criterion(temp_pred, y_val)
        
        return inner_loss, meta_loss
    
    def validate(self, model):
        model.eval()

        total_loss = 0
        total_dot_loss = 0        
        outputs = torch.Tensor().cuda()
        r2_res = 0
        with torch.no_grad():
            for wk_valid_loader in self.test_dataloaders:
                for data, target in wk_valid_loader:
                    output = model(data)
                    if self.dot:
                        d = torch.bmm(model.res_x, model.res_x.transpose(1, 2))
                        dot_loss = F.mse_loss(d, torch.eye(d.size(1)).repeat(data.shape[0], 1, 1).cuda())
                        loss = (1-self.lamb)*F.mse_loss(output, target) + self.lamb*dot_loss
                    else:
                        dot_loss = 0
                        loss = F.mse_loss(output, target)
                    true = target.cpu().detach().numpy().squeeze()
                    pred = output.cpu().detach().numpy().squeeze()
                    r2_res += r2_score(true, pred)
                    total_loss += loss.item()
                    total_dot_loss += dot_loss.item()
                    outputs = torch.cat((outputs, output))
        total_loss /= len(wk_valid_loader) * len(self.test_dataloaders)
        total_dot_loss /= len(wk_valid_loader) * len(self.test_dataloaders)
        r2_res /= len(wk_valid_loader) * len(self.test_dataloaders)

        return total_loss, outputs, r2_res
    

class adaptation_trainer():
    def __init__(self, model, train_dataloaders, test_dataloaders, num_epochs, lr, dot=True, lamb=0.6):
                        
        self.patience = 10  # for early stopping 
        self.model = model        
        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = test_dataloaders

        self.num_epochs = num_epochs
        self.lr = lr
        self.dot = dot
        self.lamb = lamb
        
    def forward(self):
        self.fit()

    def fit(self):
      
        best_loss = np.inf
        self.logger.info(f"[Train MODE] Training Model") 
        name = get_filename('model_save', 'model', '.pt')
        
        for epoch in range(self.epochs):
            loss_tr, dot_loss_tr, _ = self.train(self.model, self.train_dataloaders)
            loss_te, dot_loss_te, te_outputs, r2_res = self.valid(self.model, self.test_dataloaders)

            time = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            if self.dot:
                self.logger.info(f"[{epoch:02d}/{self.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f} \tdot loss_tr: {dot_loss_tr:.8f}\tdot loss_te:{dot_loss_te:.8f}\t r2:{r2_res:.4f}")
            else:
                self.logger.info(f"[{epoch:02d}/{self.epochs}] loss_tr: {loss_tr:.8f}\tloss_te:{loss_te:.8f}\t r2:{r2_res:.4f}")
                
            # early stopping
            if loss_te > self.best_loss:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print('Early stopping! \n training step finish')
                    breaker = True

            elif best_loss > loss_te:
                best_loss = loss_te
                self.best_model = self.model
                self.best_outputs = te_outputs
                self.best_epoch = epoch
                torch.save(self.best_model, os.path.join('model_save', name))   # save best model
        self.logger.info(f"best loss is {best_loss:.4f} in [{self.best_epoch}/{self.epochs}]epoch, save model to {os.path.join('model_save', name)}")                
        print(f"best loss is {best_loss:.4f} in [{self.best_epoch}/{self.epochs}]epoch")

        return self.best_outputs
    
    def predict(self, X):
        return self.model(X)
    
    def train(self, model, train_loader):        
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        model.train()

        total_loss = 0.
        total_dot_loss = 0.
        outputs = torch.Tensor().cuda()