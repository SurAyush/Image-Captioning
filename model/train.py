from Config import Config
from model import CaptionModel
from caption_dataset import Caption_Dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = "./image_embeddings/final_embeddings.pkl"

model = CaptionModel(config).to(device)
model.train()

dataset = Caption_Dataset(file_path, prefix_length = config.prefix_length, extract_from_file = False)     # extract_from_file = True if captions are already tokenised and stored in a .pkl file
train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
optimizer = AdamW(model.parameters(), lr=config.lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=config.epochs * len(train_loader))


for epoch in range(config.epochs):
        print(f"Epoch: {epoch}")
        progress = tqdm(total=len(train_loader), desc="Training")

        for idx, (tokens, mask, prefix) in enumerate(train_loader):

            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)      # (B, prefix_length + max_seq_length, d_model)
            logits = outputs.logits[:, config.prefix_length - 1:-1]             # (B, max_seq_length, vocab_size)
            # only consider the output for caption tokens

            # calculating loss
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)    # ignore_index to ignore loss for padding tokens
            loss.backward()
            optimizer.step()       # gradient descent
            scheduler.step()       # lr decay

            optimizer.zero_grad()
            
            # updating progress bar
            progress.set_postfix(loss=loss.item())
            progress.update()
            

        progress.close()
        
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")