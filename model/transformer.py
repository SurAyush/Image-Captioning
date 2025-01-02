import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    '''Feed Forward Network for Transformer'''

    def __init__(self, d_model, d_ff, dropout_ratio = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MultiHeadSA(nn.Module):
    '''Multi Head Self Attention Layer'''

    def __init__(self, n_heads, d_model, input_dim):  
        super().__init__()    
        assert d_model % n_heads == 0 , "Invalid head_size for the given d_model"
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_size = d_model // n_heads
        self.input_dim = input_dim
        self.qkv_proj = nn.Linear(input_dim, 3 * d_model)
        self.linear = nn.Linear(d_model, d_model)
    
    def forward(self, X, mask = None):

        B, T, C = X.shape
        assert C == self.input_dim, "Input dimension does not match the model input dimension"
        qkv = self.qkv_proj(X)                                    # (B,T,3*D)
        qkv = qkv.reshape(B, T, self.n_heads, 3 * self.d_model // self.n_heads)
        qkv = qkv.permute(0,2,1,3)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        if mask is None:
            attention_score = torch.softmax(q @ k.transpose(-2, -1) / (self.head_size ** 0.5), dim=-1)
        else:
            mask = mask.unsqueeze(1)  # for broadcasting
            attention_score = torch.softmax(q @ k.transpose(-2, -1) / (self.head_size ** 0.5) + mask, dim=-1)
        res = attention_score @ v                                       # (B,H,T,head_size)
        res = res.permute(0,2,1,3).reshape(B, T, self.d_model)   
        res = self.linear(res)

        return res               

class EncoderLayer(nn.Module):
    '''Single Layer of Transformer Encoder'''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.multi_head_sa = MultiHeadSA(self.config.n_heads, self.config.d_model, self.config.d_model)
        self.feed_forward = FeedForward(self.config.d_model, self.config.d_ff, self.config.dropout)
        self.norm1 = nn.LayerNorm(self.config.d_model)
        self.norm2 = nn.LayerNorm(self.config.d_model)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x, mask = None):
        # ordering of layernorm is like the GPT2 paper and not like the original transformer paper
        # layer norm before attention and feed forward
        res = self.norm1(x)
        res = self.multi_head_sa(res, mask)
        res = x + self.dropout(res)          # residual connection and dropout
        res = self.norm2(res)
        res2 = self.feed_forward(res)
        res = res + self.dropout(res2)

        return res

class Transformer(nn.Module):
    '''Modified Encoder Only Representation of Transformer
        for converting CLIP embedding to GPT2 input'''
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(self.config.n_layers)])

    def forward(self, x, mask = None):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        return x
    
