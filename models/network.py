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

    def forward(self, x):
        wk = x[:, -self.wk_vec_dim:] # workload vector
        x = x[:, :-self.wk_vec_dim] # knobs vector
        
        self.embed_wk = self.embedding(wk) # (batch, 4) -> (batch, dim)
        self.embed_wk = self.embed_wk.unsqueeze(1) # (batch, 1, dim)
        self.x = self.knob_fc(x) # (batch, 22) -> (batch, group*hidden)
        self.res_x = torch.reshape(self.x, (-1, self.group_dim, self.hidden_dim)) # (batch, group, hidden)
        
        self.attn_output, self.attn_weights = self.attention(self.embed_wk.permute((1,0,2)), self.res_x.permute((1,0,2)), self.res_x.permute((1,0,2)))
        self.attn_output = self.activate(self.attn_output.squeeze())
        outs = self.attn_output
        self.outputs = self.fc(outs)  
        return self.outputs
    
    def forward(self, x):
        wk = x[:, -self.wk_vec_dim:] # workload vector
        x = x[:, :-self.wk_vec_dim] # knobs vector
        
        self.embed_wk = self.embedding(wk) # (batch, 4) -> (batch, dim)
        self.embed_wk = self.embed_wk.unsqueeze(1) # (batch, 1, dim)
        self.x = self.knob_fc(x) # (batch, 22) -> (batch, group*hidden)
        self.res_x = torch.reshape(self.x, (-1, self.group_dim, self.hidden_dim)) # (batch, group, hidden)
        
        self.attn_output, self.attn_weights = self.attention(self.embed_wk.permute((1,0,2)), self.res_x.permute((1,0,2)), self.res_x.permute((1,0,2)))
        self.attn_output = self.activate(self.attn_output.squeeze())
        outs = self.attn_output
        self.outputs = self.fc(outs)  
        return self.outputs