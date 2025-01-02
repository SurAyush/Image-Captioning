import torch
import torch.nn as nn
from torch.nn import functional as F

class Transformer(nn.Module):
    pass

class Mapping_Network(nn.Module):

    def __init__(self, config, dropout=0.1):

        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.n_clip_emb, config.clip_length * config.d_model)
        self.prefix_const = nn.Parameter(torch.randn(self.prefix_length, self.d_model), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(config)


    def forward(self, x):
        pass