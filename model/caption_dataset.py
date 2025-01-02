import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import pickle

class Caption_Dataset(Dataset):

    def __init__(self, file_path, prefix_length, extract_from_file = False):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')      # gpt2 tokenizer from HF
        self.prefix_length = prefix_length
        
        # clip embedding & captions file 
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.prefixes = data["clip_embedding"]
        self.captions = data["captions"]

        self.captions_tokens = []        # to store list of tokenised captions
        max_seq_len = 0

        if extract_from_file:          # if tokenisation is already done, load from file
            with open(f"./image_embeddings/caption_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.max_seq_len = pickle.load(f)
        else:
            for caption in self.captions:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64))           # storing tokenised captions
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])    

            with open(f"./image_embeddings/caption_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, max_seq_len], f)
            
            self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.captions)
    
    def pad_mask(self,idx):

        tokens = self.captions_tokens[idx]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))          # padding with -1 tokens
            self.captions_tokens[idx] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]          # trimming the tokens
            self.captions_tokens[idx] = tokens

        mask = tokens.ge(0)              # assigns false to -1 tokens  
        tokens[~mask] = 0                # assigns 0 to false (-1) 
        mask = mask.float()
        # prefix should always be considered in attention mechanism
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)     # adding prefix length to mask
        
        return tokens, mask

    def __getitem__(self, idx):
        tokens, mask = self.pad_mask(idx)
        return tokens, mask, self.prefixes[idx]
        

        
        