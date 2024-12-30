# pip install the libraries
import time
import pickle
import torch
import skimage.io as io
import json
from PIL import Image
# huggingface
from transformers import CLIPProcessor, CLIPModel

# paths: used Kaggle paths for the dataset
out_path = ""
captions_dir = ""
img_files_loc = ""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(captions_dir,'r') as f:
    data = json.load(f) 

# loading the CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# vit-base-patch32 is the model for 512 image features

def get_img_loc(img_id):
    img_id = str(img_id)
    padded_id = img_id.rjust(12,'0')
    filename = img_files_loc + padded_id + '.jpg'
    return filename


# parsing code
image_captions = []
prefix = []                # clip outputs
ct = 0
start = time.time()

for el in data['annotations']:
    
    img_id = el['image_id']
    caption = el['caption']
    ct += 1
    
    filename = get_img_loc(img_id)
    image = io.imread(filename)                       # converts image to numpy array
    pil_img = Image.fromarray(image)                  # convert to PIL image    
    processed_img = preprocess(images = pil_img, return_tensors='pt', padding=True)        # preprocess the image
    image_tensor = processed_img['pixel_values']
    image = image_tensor.to(device)

    with torch.no_grad():
        out = model.get_image_features(pixel_values = image)

    image_captions.append(caption)
    prefix.append(out)

    if ct % 50000 == 0 :             # saving for safety purposes
        print('Checkpoint: ',ct)
        with open(out_path + str(ct) + '.pkl', 'wb') as f:
            pickle.dump({"clip_embedding": torch.cat(prefix, dim=0), "captions": image_captions}, f)


end = time.time()
print((end-start)/60)
with open(out_path + 'final_embeddings' + '.pkl', 'wb') as f:
    pickle.dump({"clip_embedding": torch.cat(prefix, dim=0), "captions": image_captions}, f)