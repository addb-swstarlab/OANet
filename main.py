import os
import argparse
import pandas as pd
import numpy as np

from models.train import train_Net
from models.utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


os.system('clear')

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, help='Choose target workload')
parser.add_argument('--external', type=str, default='RATE', help='choose which external matrix be used as performance indicator')

parser.add_argument('--batch_size', type=int, default=64, help='Define batch size of MAML step')
parser.add_argument('--in_lr', type=float, default=0.01, help='Define learning rate')  
parser.add_argument('--lr', type=float, default=0.0001, help='Define learning rate')  
parser.add_argument('--epochs', type=int, default=30, help='Define train epochs')   

parser.add_argument('--mode', type=str, default='reshape', help='choose which model be used on fitness function')
parser.add_argument('--hidden_size', type=int, default=16, help='Define model hidden size')
parser.add_argument('--group_size', type=int, default=32, help='Define model gruop size')
parser.add_argument('--dot', action='store_true', help='if trigger, model use loss term, dot')
parser.add_argument('--lamb', type=float, default=0.6, help='define lambda of loss function' )

parser.add_argument('--train', action='store_true', help='if trigger, model goes triain mode')
parser.add_argument('--eval', action='store_true', help='if trigger, model goes eval mode')




opt = parser.parse_args()

if not os.path.exists('logs'):
    os.mkdir('logs')

logger, log_dir = get_logger(os.path.join('./logs'))


logger.info("## model hyperparameter information ##")
for i in vars(opt):
    logger.info(f'{i}: {vars(opt)[i]}')

DATA_PATH = "./data"
WK_NUM = 13
EX_NUM = 4


def main():
    
    logger.info("## get raw datas ##")

    X_tr, X_te = []   # train data of knob
    Y_tr, Y_te = []   # train data of external metric
    one_hot = np.eye(WK_NUM)
    
    for wk in range(WK_NUM):
        if wk == (WK_NUM - 1):
            raw_data_wk = pd.read_csv(os.path.join(DATA_PATH, f'rocksdb_benchmark_{str(opt.target)}.csv'))
        else:
            raw_data_wk = pd.read_csv(os.path.join(DATA_PATH, f'rocksdb_benchmark_{str(wk)}.csv'))
            
        oh = np.repeat([one_hot[wk]], len(raw_data_wk), axis=0)
        oh = pd.DataFrame(oh)
        raw_data_wk = pd.concat((raw_data_wk, oh), axis=1)
        
        X = raw_data_wk.iloc[:,EX_NUM:]
        Y = raw_data_wk[[opt.external]]       
        x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2, shuffle=True)
        X_tr.append(x_tr)
        X_te.append(x_te)
        Y_tr.append(y_tr)
        Y_te.append(y_te)
        
    logger.info('## get raw data DONE ##')

    # calculate scaler
    X_tr_concat = pd.concat(X_tr)
    Y_tr_concat = pd.concat(Y_tr)    
    scaler_X = MinMaxScaler().fit(X_tr_concat)
    scaler_Y = StandardScaler().fit(Y_tr_concat)     

    # make dataloader
    logger.info('## make dataloader ##')
     
    DL_tr, DL_te, norm_x_te, norm_y_te = MAML_dataset(X_tr, Y_tr, X_te, Y_te, scaler_X, scaler_Y, WK_NUM, batch_size=opt.batch_size)
    DL_val, _, _, _  = MAML_dataset(X_tr, Y_tr, X_te, Y_te, scaler_X, scaler_Y, WK_NUM, batch_size=opt.batch_size)    
    
    logger.info('## make dataloader DONE ##')

    
    if opt.train:
        pass
            
    elif opt.eval:
        logger.info('## EVAL MODE ##')

if __name__ == '__main__':
    try:
        main()
    except:
        logger.exception("ERROR!!")
    finally:
        logger.handlers.clear()