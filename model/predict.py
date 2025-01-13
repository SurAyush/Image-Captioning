import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, CLIPProcessor, CLIPModel
import skimage.io as io
import PIL.Image
from model.Config import Config
from model.model import CaptionModel

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Predictor():
    
    def __init__(self, path, config = config):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)                   # CLIP Model
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", return_tensor='pt')      # CLIP Processor
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")                                                   # GPT2 Tokenizer
        self.config = config
        self.model = CaptionModel(self.config)                                                                   # Caption Model (trained)
        trained_state_dict = torch.load(path, map_location= torch.device('cpu'),  weights_only=True)

        # as it is saved using nn.DataParallel, we need to remove the 'module.' prefix from the keys
        updated_state_dict = {k.replace("module.", ""): v for k, v in trained_state_dict.items()}

        self.model.load_state_dict(updated_state_dict)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    
    def predict(self, image, beam = False, beam_size = 5):
        
        image = io.imread(image)
        pil_img = PIL.Image.fromarray(image)
        processed_img = self.preprocess(images = pil_img, return_tensors='pt', padding=True)
        image_tensor = processed_img['pixel_values']
        image = image_tensor.to(self.device)
        
        with torch.no_grad():
            prefix = self.clip_model.get_image_features(pixel_values = image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_embedding_mapping(prefix).reshape(1, self.config.prefix_length, -1)
        
        if beam == False:
            return generate(self.model, self.tokenizer, embed=prefix_embed)
        else:
            return generate_beam(self.model, self.tokenizer, embed=prefix_embed, beam_size=beam_size)
    

# simple greedy decoding using top-k sampling and multinomial sampling
def generate(model,tokenizer,embed):

    model.eval()
    stop_token = '.'                                                       # stop token for the generated caption
    stop_token2 = '\n'
    stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token2_index = tokenizer.encode(stop_token2)[0]
    max_length= 45                                                         # maximum length of the generated caption
    tokens = None
    top_k = 10                                                             # top-k sampling for the next token generation
    

    with torch.no_grad():
        
        generated = embed
        
        for _ in range(max_length):

            outputs = model.gpt(inputs_embeds=generated)                                        # embedding of image generated 
            logits = outputs.logits
            logits = logits[:, -1, :]                                                           # taking the last token for auto-regressive generation

            # using top-k and multinomial for the next token generation
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1)                   # top-k sampling
            probs = F.softmax(top_k_logits, dim=-1)       
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(dim=-1, index=next_token_idx)                     # Map sampled token index back to the original vocabulary
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token                                                             # initialization 
            else:
                tokens = torch.cat((tokens, next_token), dim=1)                                 # concatenating the generated tokens

            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item() or stop_token2_index == next_token.item():
                break

        output_list = list(tokens.squeeze().cpu().numpy())                                      # removing the batch dimension, converting to numpy and then to list
        output_text = tokenizer.decode(output_list)
        
        
    return [output_text]


# beam search decoding
def generate_beam(model,tokenizer,embed, beam_size = 5):

    model.eval()
    stop_token = '.'
    stop_token2 = '\n'
    stop_token_index = tokenizer.encode(stop_token)[0]
    stop_token2_index = tokenizer.encode(stop_token2)[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    entry_length = 45
    tokens = None
    scores = None                                                                         # 1d tensor containing score of every prediction
    seq_lengths = torch.ones(beam_size, device=device)                                    # contains seq len of each prediction
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)                  # flags for early stopping
    
    with torch.no_grad():
            
        generated = embed                                                                  # (1,prefix_length=20,d_model=768)
            
        for _ in range(entry_length):

            outputs = model.gpt(inputs_embeds=generated)                                    # initial embedding of image generated
            logits = outputs.logits
            logits = logits[:, -1, :]                                                       # taking the last token for auto-regressive generation
            probs = F.softmax(logits, dim=-1)
            out = probs.log()                                                               # converting probabilities to log-probabilities (better for scoring)
            # out: (1,vocab_size) -> initially , later (beam_size, vocab_size)

            if scores is None:                                                             
                scores, next_tokens = out.topk(beam_size, -1)                                             # scores(1,beam_size): out , next_tokens(1,beam_size): indices
                generated = generated.expand(beam_size, generated.shape[1], generated.shape[2])           # expanding over beam_size
                next_tokens = next_tokens.permute(1, 0)                                                   # making B predictions with top-b tokens
                scores = scores.squeeze(0)                                                                # removing batch-dim
                tokens = next_tokens
                
            else:
                logits[is_stopped] = -float(np.inf)                                                      # marks all values as -inf for every beam which is stopped
                logits[is_stopped, 0] = 0                                                                # initial element of every stopped search as 0

                # scores[:,None] is same as scores.reshape(beam_size,1) but it doesn't occupy extra memory
                scores_sum = scores[:, None] + logits                                                    # scores is reshaped from 1d tensor to 2d for broadcasted addition
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]                                   # avg: better judging parameter for variable sized outputs
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)        # re-evalutaing and finding top k (beam-size) logits
                    
                next_tokens_source = next_tokens // scores_sum.shape[1]                                # indices are flattened, mapping to corresponding beam
                seq_lengths = seq_lengths[next_tokens_source]                                          
                next_tokens = next_tokens % scores_sum.shape[1]                                        # flattened index, mapping back to original index to tokens
                next_tokens = next_tokens.unsqueeze(1)

                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
                
                
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)                # auto-regressive generation
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index or stop_token2_index).squeeze()                                                # checking if stop token is generated
            
            if is_stopped.all():
                break
                    
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[: int(length)]) for output, length in zip(output_list, seq_lengths)]
        
    ordered = scores.argsort(descending=True)                 # best to worst
    output_texts = [output_texts[i] for i in ordered]
        
    return output_texts
                


if __name__ == '__main__':
    
    path = "./trained_model/model_epoch_4.pt"
    image = "image.jpg"
    predictor = Predictor(path, config)
    print(predictor.predict(image))