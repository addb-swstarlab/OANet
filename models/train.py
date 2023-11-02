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
    
    model = ReshapeNet(input_dim=norm_x_te.shape[-1], hidden_dim=hidden_dim, output_dim=norm_y_te.shape[-1], group_dim=group_dim, wk_vec_dim=WK_NUM, dot=dot, lamb=lamb).cuda()

    # MAML step
    maml = MAML_trainer(model, MAML_DL_tr, MAML_DL_val, MAML_DL_te, num_epochs=epochs, inner_lr=lr, meta_lr=in_lr)
    maml.main_loop()
    trained_MAML_pt_path = maml.name
    model.load_state_dict(torch.load(trained_MAML_pt_path))
    
    # Adaptation step
    adapt = adaptation_trainer(model, Adapt_DL_tr, Adapt_DL_te, num_epochs=epochs, lr=lr)
    adapt.fit()
    
    # Predict
    outputs = adapt.predict(norm_x_te)

    true = norm_y_te.cpu().detach().numpy().squeeze()
    pred = outputs.cpu().detach().numpy().squeeze()
    
    R2 = r2_score(true, pred)
    MSE = mean_squared_error(true, pred)

    del norm_X_tr, norm_X_te, norm_y_tr, norm_y_te
    torch.cuda.empty_cache()
    
    df_pred = pd.DataFrame(columns=("METRIC", "R2", "MSE"))    
    score = [ (METRIC, R2, MSE) ]
    ex = pd.DataFrame(score, columns=["METRIC", "R2", "MSE"])
    df_pred = pd.concat(([df_pred, ex]), ignore_index=True )
    return R2, MSE, true, pred, df_pred