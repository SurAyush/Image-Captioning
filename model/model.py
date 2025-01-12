# Transformer based mapping model to map image embeddings to input to GPT-2
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from model.mapping_network import Mapping_Network

class CaptionModel(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.config = config
        self.clip_embedding_mapping = Mapping_Network(config)

    def forward(self, tokens, prefix, mask):
        cap_emb = self.gpt.transformer.wte(tokens)          # (batch_size, seq_len, embedding_size)
        clip_emb = self.clip_embedding_mapping(prefix).view(-1,self.config.prefix_length,self.gpt_embedding_size)      # (batch_size, prefix_length, d_model_gpt2)
        res = torch.cat((clip_emb, cap_emb), dim=1)
        res = self.gpt(inputs_embeds = res, attention_mask = mask)

        return res
    
    def train(self, mode = True):
        super(CaptionModel, self).train(mode)
        # freeze and train
        self.gpt.eval()                      # gpt2 weights remain fixed
        return self
