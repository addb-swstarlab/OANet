import torch
import torch.nn as nn
import torch.nn.functional as F


class ReshapeNet(nn.Module):
    def __init__(self, input_dim, wk_vec_dim, output_dim, hidden_dim=16, group_dim=32):
        super(ReshapeNet, self).__init__()
        self.input_dim = input_dim - wk_vec_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.group_dim = group_dim
        self.wk_vec_dim = wk_vec_dim

        self.embedding = nn.Linear(self.wk_vec_dim, self.hidden_dim)
        self.knob_fc = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim*self.group_dim), nn.ReLU()) # (22, 1) -> (group*hidden, 1)
        self.attention = nn.MultiheadAttention(self.hidden_dim, 1)
        self.activate = nn.ReLU()
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
