import argparse
import copy
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pydash import at
#from pydash import at
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torchvision.transforms.functional import to_pil_image

class RadReStructPrecomputed(Dataset):
    """
    dataset for question-level training and inference: each element consists of the image and one corresponding question with history

        returns:
            img: transformed image tensor
            tokens: tokenized question + history
            q_attn_mask: attention mask for question + history
            attn_mask: attention mask for image + question + history
            answer: multi-hot encoding of label vector
            token_type_ids: token type ids for question + history
            info: dict with additional information about the question (single or multiple choice, options)
            mask: label mask for loss calculation (which answer options are valid for this question)
    """

    def __init__(self, tfm=None, mode='train', args=None, precompute=False):

        self.tfm = tfm
        self.args = args
        self.precompute = precompute

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)
        
        if precompute:
            with open(f'/home/guests/adrian_delchev/code/ad_Rad-ReStruct/precomputed/trainvaltest_image_embeddings_f32.pkl', 'rb') as f:
                self.precomputed_image_embeddings = pickle.load(f)
            with open(f'/home/guests/adrian_delchev/code/ad_Rad-ReStruct/precomputed/trainvaltest_global_embeddings_f32.pkl', 'rb') as f:
                self.precomputed_global_embeddings = pickle.load(f)



        # get file names of data in radrestruct/{split}_qa_pairs
        with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
            self.id_to_img_mapping = json.load(f)

        self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        # all images corresponding to reports in radrestruct/{split}_qa_pairs
        self.images = []
        for report in self.reports:
            for elem in self.id_to_img_mapping[report.split('.')[0]]:
                self.images.append((report, elem))

        # get all question - image pairs
        self.samples = []
        for report, img in self.images:
            report_id = report.split('.')[0]
            with open(f'data/radrestruct/{mode}_qa_pairs/{report}', 'r') as f:
                qa_pairs = json.load(f)

            for qa_pair_idx, qa_pair in enumerate(qa_pairs):
                sample_id = f"{img}_{report_id}_{qa_pair_idx}"
                self.samples.append((qa_pair, img, sample_id))

        self.mode = mode
        with open('data/radrestruct/answer_options.json', 'r') as f:
            self.answer_options = json.load(f)

        #self.samples = self.samples[2:3] # For debugging of validation, comment out after

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qa_sample, img_name, sample_id = self.samples[idx]

        if self.precompute:
            #img = self.precomputed_image_embeddings[img_name]
            img = torch.from_numpy(self.precomputed_image_embeddings[img_name])
            global_embed = torch.from_numpy(self.precomputed_global_embeddings[img_name])
        else:
            images_path = os.path.join(self.args.data_dir, 'images')

            img_path = Path(images_path) / f'{img_name}.png'
            img = Image.open(img_path)
            #img.save("/home/guests/adrian_delchev/preview_images/rr_pre_transform.jpeg")
            if self.tfm:
                img = self.tfm(img)
                #transformed_image = to_pil_image(img)
                #transformed_image.save("/home/guests/adrian_delchev/preview_images/rr_post_transform.jpeg")

        question = qa_sample[0]
        answer = qa_sample[1]
        history = qa_sample[2]
        info = qa_sample[3]
        info['positive_option'] = answer

        ### info contains:
        ### answer_type: str = 'single_choice'/'multiple_choice'
        ### options: list(str): ['yes', 'no']
        ### path: str: 'foreign_objects'
        ### positive_option: list(str): ['no']
        
        # create loss mask
        options = info['options']
        answer_idxs = at(self.answer_options, *options)
        # create mask of size (batch_size, num_options) where 1 means that the option idx is in answer_idxs for this batch element
        mask = torch.zeros((len(self.answer_options)), dtype=torch.bool)
        mask[answer_idxs] = 1
        answer = encode_answer(answer, self.answer_options)
        
        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, self.tokenizer, mode=self.mode, args=self.args)
        
        return (img, global_embed), torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
               torch.tensor(answer, dtype=torch.float), token_type_ids, info, mask