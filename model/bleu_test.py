import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import random
from collections import defaultdict
from model.predict import Predictor

nltk.download('punkt')

dataset_dir = ''        # path to the dataset directory (we used Kaggle)
img_files_loc = ''      # path to the images directory

class BLEU_Score:

    def __init__(self,n_samples, captions_dir):
        
        # make dataset from MS COCO sampling
        with open(captions_dir,'r') as f:
            data = json.load(f)

        # create dataset of img_id and captions
        image_captions = defaultdict(list)

        for el in data['annotations']:
            img_id = el['image_id']  
            caption = el['caption']  
            image_captions[img_id].append(caption)

        random_keys = random.sample(list(image_captions.keys()), n_samples)
        imgid_captions = {key: image_captions[key] for key in random_keys}

        # predict using model
        self.predictor = Predictor()

        self.predictions = []
        self.reference_captions = []

        for key,value in imgid_captions.items():
            img = self.get_img_loc(key)
            self.predictions.append(self.predict(img))
            self.reference_captions.append(value)


    def get_img_loc(img_id):
        img_id = str(img_id)
        padded_id = img_id.rjust(12,'0')
        filename = img_files_loc + padded_id + '.jpg'
        return filename

    def get_bleu_score(self):
        res = self.calculate_bleu(self.reference_captions, self.predictions)
        return res

    def calculate_bleu(reference_captions, generated_captions):
        
        smooth_fn = SmoothingFunction().method1
        
        # Calculate BLEU scores
        bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
        for refs, gen in zip(reference_captions, generated_captions):
            # Tokenize captions
            refs_tokenized = [nltk.word_tokenize(ref) for ref in refs]
            gen_tokenized = nltk.word_tokenize(gen)
            
            # Compute BLEU scores for this caption
            bleu1 += sentence_bleu(refs_tokenized, gen_tokenized, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
            bleu2 += sentence_bleu(refs_tokenized, gen_tokenized, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
            bleu3 += sentence_bleu(refs_tokenized, gen_tokenized, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn)
            bleu4 += sentence_bleu(refs_tokenized, gen_tokenized, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        
        # Average BLEU scores over all captions
        total = len(generated_captions)
        return {
            "BLEU-1": bleu1 / total,
            "BLEU-2": bleu2 / total,
            "BLEU-3": bleu3 / total,
            "BLEU-4": bleu4 / total
        }


n_samples = 5000

bleu = BLEU_Score(n_samples, dataset_dir)
output = bleu.get_bleu_score()

with open('bleu_output.json', 'w') as f:
    json.dump(output, f, indent=3)