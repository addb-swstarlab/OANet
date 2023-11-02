import torch
import pandas as pd
import numpy as np
from models.network import ReshapeNet
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from models.neural_network_train import MAML_trainer, adaptation_trainer

def train_Net(DL_tr, DL_val, DL_te, norm_x_te, norm_y_te, WK_NUM, opt):
    METRIC = opt.external
    MODE = opt.mode
    in_lr = opt.in_lr
    lr = opt.lr
    epochs = opt.epochs
    hidden_dim = opt.hidden_size
    group_dim = opt.group_size
    dot = opt.dot
    lamb=opt.lamb

    # Split DL_tr, DL_te to MAML, Adaptation dataloader
    MAML_DL_tr, MAML_DL_val, MAML_DL_te = DL_tr[:-1], DL_val[:-1], DL_te[:-1]
    Adapt_DL_tr, Adapt_DL_val, Adapt_DL_te = DL_tr[-1], DL_val[-1], DL_te[-1]