import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, CLIPProcessor, CLIPModel
import skimage.io as io
import PIL.Image
from Config import Config
from model import CaptionModel

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Predictor():
    
    def __init__(self, config, path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)         # CLIP Model
        self.preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", return_tensor='pt')     # CLIP Processor
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")         # GPT2 Tokenizer
        self.config = config
        self.model = CaptionModel(self.config)         # Caption Model (trained)
        trained_state_dict = torch.load(path, map_location= torch.device('cpu'),  weights_only=True)

        # as it is saved using nn.DataParallel, we need to remove the 'module.' prefix from the keys
        updated_state_dict = {k.replace("module.", ""): v for k, v in trained_state_dict.items()}

        self.model.load_state_dict(updated_state_dict)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    
    def predict(self, image):
        
        image = io.imread(image)
        pil_img = PIL.Image.fromarray(image)
        processed_img = self.preprocess(images = pil_img, return_tensors='pt', padding=True)
        image_tensor = processed_img['pixel_values']
        image = image_tensor.to(self.device)
        
        with torch.no_grad():
            prefix = self.clip_model.get_image_features(pixel_values = image).to(self.device, dtype=torch.float32)
            prefix_embed = self.model.clip_embedding_mapping(prefix).reshape(1, self.config.prefix_length, -1)
        
        
        return generate(self.model, self.tokenizer, embed=prefix_embed)


def generate(model,tokenizer,embed):

    model.eval()
    stop_token = '.'     # stop token for the generated caption
    stop_token_index = tokenizer.encode(stop_token)[0]
    entry_length = 75    # maximum length of the generated caption
    tokens = None
    generated_list = []
    

    with torch.no_grad():
        
        generated = embed
        
        for _ in range(entry_length):

            outputs = model.gpt(inputs_embeds=generated)        # embedding of image generated 
            logits = outputs.logits
            logits = logits[:, -1, :]                           # taking the last token for auto-regressive generation
            probs = F.softmax(logits, dim=-1)       
            next_token = torch.multinomial(probs,1)             # sampling from the distribution of the next token      
            next_token_embed = model.gpt.transformer.wte(next_token)
            if tokens is None:
                tokens = next_token      # first time 
            else:
                tokens = torch.cat((tokens, next_token), dim=1)      # concatenating the generated tokens

            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item():
                break

        output_list = list(tokens.squeeze().cpu().numpy())           # removing the batch dimension, converting to numpy and then to list
        output_text = tokenizer.decode(output_list)
        generated_list.append(output_text)                         # to support multiple generations in future

        
    return generated_list[0]


if __name__ == '__main__':
    
    path = "./trained_model/model_epoch_4.pt"
    image = "image.jpg"
    predictor = Predictor(config, path)
    print(predictor.predict(image))