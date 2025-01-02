import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer import Transformer

class Mapping_Network(nn.Module):

    def __init__(self, config, dropout=0.1):

        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.n_clip_emb, config.clip_length * config.d_model)        # 512 -> clip_length * d_model(768)
        self.fixed_prefix = nn.Parameter(torch.randn(self.prefix_length, self.d_model), requires_grad=True)            # fixed prefix
        self.transformer = Transformer(config)


    def forward(self, x):
        # x: (batch_size, n_clip_emb)
        res = self.linear(x)          # (batch_size, clip_length * d_model)
        res = res.view(res.shape[0], self.config.clip_length, self.config.d_model)        # (batch_size, clip_length, d_model)
        prefix = self.fixed_prefix.unsqueeze(0)             # adding batch dimension
        prefix = prefix.repeat(res.shape[0], 1, 1)          # (batch_size, prefix_length, d_model)
        # first clip_embedding followed by fixed prefix
        res = torch.cat((res, prefix), dim=1)               # (batch_size, prefix_length + clip_length, d_model)
        res = self.transformer(res)                         
        
        return res[:,self.config.clip_length:]       
