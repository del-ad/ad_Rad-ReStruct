import argparse
import copy
import json
import os
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


# Original 4 tokens version
def encode_text_progressive(question, history, tokenizer, mode, args):
    encoded_history = []
    token_type_ids = []

    if mode == "train" and args.aug_history:  # augmentation: random dropping of lvl 1 and lvl 3 history questions, random order of lvl 3 questions
        if len(history) > 0:
            keep_num = int(torch.randint(0, len(history) + 1, (1,)))  # drops between 0 and len(history) elements
            # randomly sample keep_num elements from history
            keep_idxs = random.sample(range(len(history)), keep_num)
            # always keep level 2 question as it shows the current position in the report
            if len(history) > 1 and 1 not in keep_idxs:
                keep_idxs.append(1)

            keep_idxs = sorted(keep_idxs)
            history = [history[i] for i in keep_idxs]

            # randomly shuffle level 3 questions
            if len(history) > 2:
                # find idx of question id 1 in keep_idxs
                lvl2_idx = keep_idxs.index(1)

                # shuffle all after lvl2_idx
                copy = history[lvl2_idx + 1:]
                random.shuffle(copy)
                history[lvl2_idx + 1:] = copy

    if args.progressive:  # else leave history empty
        for elem in history:
            encoded_question = tokenizer.encode(elem[0])[1:-1]
            # sanity check to see if above comment is correct
            #encoded_question_w_specials = tokenizer.encode(elem[0])
            # just to see what's going on - COMMENT out when done inspecting
            #dec_q_orig = tokenizer.decode(encoded_question)
            #dec_q_mine = tokenizer.decode(encoded_question_w_specials)

            # what happens when there are multiple answers ? (len(elem[1]>1))
            encoded_answer = tokenizer.encode(', '.join(elem[1]))[1:-1]  # list of all answers to the question
            encoded_history.append(encoded_question + [tokenizer.sep_token_id] + encoded_answer + [tokenizer.sep_token_id])
            token_type_ids.append([2] * (len(encoded_question) + 1) + [3] * (len(encoded_answer) + 1))
            # COMMENT out when done
            #ncoded_answer_w_specials = tokenizer.encode(', '.join(elem[1]))  # list of all answers to the question
            #dec_a_orig = tokenizer.decode(encoded_answer)
            #dec_a_mine = tokenizer.decode(ncoded_answer_w_specials)
    # add current question
    encoded_question = tokenizer.encode(question)[1:-1]
    encoded_history.append(encoded_question)
    token_type_ids.append([1] * len(encoded_question))  # current question is always type 1

    # flatten
    encoded_history = [item for sublist in encoded_history for item in sublist]
    token_type_ids = [item for sublist in token_type_ids for item in sublist]

    # in gt this is never exceeded
    # for now just truncate history
    if len(encoded_history) > args.num_question_tokens:
        print(f"Truncating history: {question}, {history}")
        encoded_history = encoded_history[-args.num_question_tokens:]
        token_type_ids = token_type_ids[-args.num_question_tokens:]

    assert len(encoded_history) <= args.num_question_tokens

    tokens = [tokenizer.cls_token_id] + [tokenizer.sep_token_id] + encoded_history + [tokenizer.sep_token_id]
    # Comment out when done
    #tokens_text = tokenizer.decode(tokens)
    # first [1] corresponds to [CLS] token, second to a [SEP] token, look line 81 above
    # in final token_type_id
    token_type_ids = [1] + [1] + token_type_ids + [1]
    q_attn_mask = [1] * len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    attn_mask = (args.num_image_tokens + len(tokens)) * [1] + (args.max_position_embeddings - len(tokens) - args.num_image_tokens) * [0]
    q_attn_mask.extend([0] * n_pad)
    # Padds with 0s - 0 is actually the <CLS> token, shouldn't we pad with PAD token instead?
    # tokens.extend([0] * n_pad)
    tokens.extend([tokenizer.pad_token_id] * n_pad)
    token_type_ids.extend([0] * n_pad)

    assert len(tokens) == args.max_position_embeddings
    assert len(q_attn_mask) == args.max_position_embeddings
    assert len(attn_mask) == args.max_position_embeddings
    assert len(token_type_ids) == args.max_position_embeddings

    return tokens, q_attn_mask, attn_mask, torch.tensor(token_type_ids, dtype=torch.long)

# 7 tokens version + pad token used
# def encode_text_progressive(question, history, tokenizer, mode, args):
#     encoded_history = []
#     token_type_ids = []
#
#     if mode == "train" and args.aug_history:  # augmentation: random dropping of lvl 1 and lvl 3 history questions, random order of lvl 3 questions
#         if len(history) > 0:
#             keep_num = int(torch.randint(0, len(history) + 1, (1,)))  # drops between 0 and len(history) elements
#             # randomly sample keep_num elements from history
#             keep_idxs = random.sample(range(len(history)), keep_num)
#             # always keep level 2 question as it shows the current position in the report
#             if len(history) > 1 and 1 not in keep_idxs:
#                 keep_idxs.append(1)
#
#             keep_idxs = sorted(keep_idxs)
#             history = [history[i] for i in keep_idxs]
#
#             # randomly shuffle level 3 questions
#             if len(history) > 2:
#                 # find idx of question id 1 in keep_idxs
#                 lvl2_idx = keep_idxs.index(1)
#
#                 # shuffle all after lvl2_idx
#                 copy = history[lvl2_idx + 1:]
#                 random.shuffle(copy)
#                 history[lvl2_idx + 1:] = copy
#
#     if args.progressive:  # else leave history empty
#         # try with sep and pad tokens also (6 total) 0 = CLS, 1 = sep, 2 = pad, 3 = img, 4 = historyq ,
#         # 5 = historya, 6 = currentq
#         for elem in history:
#             # with no beginning and end tokens of the question (special tokens)?
#             encoded_question = tokenizer.encode(elem[0])[1:-1]
#             # sanity check to see if above comment is correct
#             #encoded_question_w_specials = tokenizer.encode(elem[0])
#             # just to see what's going on - COMMENT out when done inspecting
#             #dec_q_orig = tokenizer.decode(encoded_question)
#             #dec_q_mine = tokenizer.decode(encoded_question_w_specials)
#
#             # what happens when there are multiple answers ? (len(elem[1]>1))
#             encoded_answer = tokenizer.encode(', '.join(elem[1]))[1:-1]  # list of all answers to the question
#             # COMMENT out when done
#             #ncoded_answer_w_specials = tokenizer.encode(', '.join(elem[1]))  # list of all answers to the question
#             #dec_a_orig = tokenizer.decode(encoded_answer)
#             #dec_a_mine = tokenizer.decode(ncoded_answer_w_specials)
#
#             encoded_history.append(
#                 encoded_question + [tokenizer.sep_token_id] + encoded_answer + [tokenizer.sep_token_id])
#             token_type_ids.append([4] * (len(encoded_question)) + [1] + [5] * (len(encoded_answer)) + [1])
#
#     # add current question
#     encoded_question = tokenizer.encode(question)[1:-1]
#     encoded_history.append(encoded_question)
#     token_type_ids.append([6] * len(encoded_question))  # current question is always type 6
#
#     # flatten
#     encoded_history = [item for sublist in encoded_history for item in sublist]
#     token_type_ids = [item for sublist in token_type_ids for item in sublist]
#
#     # in gt this is never exceeded
#     # for now just truncate history
#     if len(encoded_history) > args.num_question_tokens:
#         print(f"Truncating history: {question}, {history}")
#         encoded_history = encoded_history[-args.num_question_tokens:]
#         token_type_ids = token_type_ids[-args.num_question_tokens:]
#
#     assert len(encoded_history) <= args.num_question_tokens
#
#     tokens = [tokenizer.cls_token_id] + [tokenizer.sep_token_id] + encoded_history + [tokenizer.sep_token_id]
#     # Comment out when done
#     tokens_text = tokenizer.decode(tokens)
#     # first [1] corresponds to [CLS] token, second to a [SEP] token, look line 81 above
#     # in final token_type_id
#     token_type_ids = [0] + [1] + token_type_ids + [1]
#     q_attn_mask = [1] * len(tokens)
#     n_pad = args.max_position_embeddings - len(tokens)
#     attn_mask = (args.num_image_tokens + len(tokens)) * [1] + (
#                 args.max_position_embeddings - len(tokens) - args.num_image_tokens) * [0]
#     q_attn_mask.extend([0] * n_pad)
#     # Padds with 0s - 0 is actually the <CLS> token, shouldn't we pad with PAD token instead?
#     # tokens.extend([0] * n_pad)
#     tokens.extend([tokenizer.pad_token_id] * n_pad)
#     token_type_ids.extend([2] * n_pad)
#
#     assert len(tokens) == args.max_position_embeddings
#     assert len(q_attn_mask) == args.max_position_embeddings
#     assert len(attn_mask) == args.max_position_embeddings
#     assert len(token_type_ids) == args.max_position_embeddings
#
#     return tokens, q_attn_mask, attn_mask, torch.tensor(token_type_ids, dtype=torch.long)

def encode_answer(answer, answer_options):
    # convert to multi-hot encoding
    answer_labels = [0] * len(answer_options)
    for ans in answer:
        answer_labels[answer_options[ans]] = 1

    return answer_labels


def calculate_bin_weights(samples, answer_options):
    class_occurences = [0] * len(answer_options)
    for sample in samples:
        for option in sample[0][3]["options"]:
            class_occurences[answer_options[option]] += 1

    vector_answers = np.array([encode_answer(sample[0][1], answer_options) for sample in samples])
    pos_occurrences = np.sum(vector_answers, axis=0)
    pos_weights = np.divide(class_occurences, pos_occurrences)

    # save class weights
    with open('data/radrestruct/all_pos_weights.json', 'w') as f:
        json.dump(pos_weights.tolist(), f)
    return pos_weights

### Original RRS
def get_targets_for_split(split, limit_data=None):
    with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
        id_to_img_mapping = json.load(f)
    reports = sorted(os.listdir(f'data/radrestruct/{split}_vectorized_answers'))
    if limit_data is not None:
        reports = reports[:limit_data]  # for evaluation during training

    targets = []
    for report in reports:
        with open(f'data/radrestruct/{split}_vectorized_answers/{report}', 'r') as f:
            report_dict = json.load(f)

        report_vector = list(report_dict.values())
        for _ in id_to_img_mapping[report.split('.')[0]]:  # some reports are appended multiple times as they correspond to multiple images
            targets.append(report_vector)

    targets = np.array(targets, dtype=float)
    return targets

### New
# def get_targets_for_split(split, limit_data=None):
#     with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
#         id_to_img_mapping = json.load(f)
#     reports = sorted(os.listdir(f'data/radrestruct/{split}_vectorized_answers'))
#     if limit_data is not None:
#         reports = reports[:limit_data]  # for evaluation during training

#     targets = []
    
    
#     if split == 'train':
#         #reports = ['2729_reversed.json']
#         reports = ['2729.json']
    
    
#     for report in reports:
#         with open(f'data/radrestruct/{split}_vectorized_answers/{report}', 'r') as f:
#             report_dict = json.load(f)

#         report_vector = list(report_dict.values())
#         if 'reversed' in report:
#             for _ in id_to_img_mapping[report.replace("_reversed", "").split('.')[0]]:  # some reports are appended multiple times as they correspond to multiple images
#                 targets.append(report_vector)
#         else:
#             for _ in id_to_img_mapping[report.split('.')[0]]:  # some reports are appended multiple times as they correspond to multiple images
#                 targets.append(report_vector)

#     targets = np.array(targets, dtype=float)
#     ### only get 1 report
#     return targets[0:1]

### Original
class RadReStruct(Dataset):
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

    def __init__(self, tfm=None, mode='train', args=None):

        self.tfm = tfm
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

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

        self.samples = self.samples[:1] # For debugging of validation, comment out after

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        qa_sample, img_name, sample_id = self.samples[idx]

        # data_dir/images - c/user/project/data/images
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
        return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
                torch.tensor(answer, dtype=torch.float), token_type_ids, info, mask

### New
# class RadReStruct(Dataset):
#     """
#     dataset for question-level training and inference: each element consists of the image and one corresponding question with history

#         returns:
#             img: transformed image tensor
#             tokens: tokenized question + history
#             q_attn_mask: attention mask for question + history
#             attn_mask: attention mask for image + question + history
#             answer: multi-hot encoding of label vector
#             token_type_ids: token type ids for question + history
#             info: dict with additional information about the question (single or multiple choice, options)
#             mask: label mask for loss calculation (which answer options are valid for this question)
#     """

#     def __init__(self, tfm=None, mode='train', args=None):

#         self.tfm = tfm
#         self.args = args

#         self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

#         # get file names of data in radrestruct/{split}_qa_pairs
#         with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
#             self.id_to_img_mapping = json.load(f)

#         self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        
        
#         ### SINGLE OUT A REPORT
#         if mode == 'train':
#             self.reports = ['2729.json']
#             #self.reports = ['2729_reversed.json']
        
        
        
        
#         # all images corresponding to reports in radrestruct/{split}_qa_pairs
#         self.images = []
#         for report in self.reports:
#             for elem in self.id_to_img_mapping[report.split('.')[0]]:
#                 self.images.append((report, elem))

#         # get all question - image pairs
#         self.samples = []
#         for report, img in self.images:
#             report_id = report.split('.')[0]
#             with open(f'data/radrestruct/{mode}_qa_pairs/{report}', 'r') as f:
#                 qa_pairs = json.load(f)

#             for qa_pair_idx, qa_pair in enumerate(qa_pairs):
#                 sample_id = f"{img}_{report_id}_{qa_pair_idx}"
#                 self.samples.append((qa_pair, img, sample_id))

#         self.mode = mode
#         with open('data/radrestruct/answer_options.json', 'r') as f:
#             self.answer_options = json.load(f)

#         # how many QA pairs
#         #self.samples = self.samples[3:4] # For debugging of validation, comment out after
#         self.samples = self.samples[0:1]

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         qa_sample, img_name, sample_id = self.samples[idx]

#         # data_dir/images - c/user/project/data/images
#         images_path = os.path.join(self.args.data_dir, 'images')

#         img_path = Path(images_path) / f'{img_name}.png'
#         img = Image.open(img_path)
#         #img.save("/home/guests/adrian_delchev/preview_images/rr_pre_transform.jpeg")
#         if self.tfm:
#             img = self.tfm(img)
#             #transformed_image = to_pil_image(img)
#             #transformed_image.save("/home/guests/adrian_delchev/preview_images/rr_post_transform.jpeg")

#         question = qa_sample[0]
#         answer = qa_sample[1]
#         history = qa_sample[2]
#         info = qa_sample[3]
#         info['positive_option'] = answer

#         ### info contains:
#         ### answer_type: str = 'single_choice'/'multiple_choice'
#         ### options: list(str): ['yes', 'no']
#         ### path: str: 'foreign_objects'
#         ### positive_option: list(str): ['no']
        
#         # create loss mask
#         options = info['options']
#         answer_idxs = at(self.answer_options, *options)
#         # create mask of size (batch_size, num_options) where 1 means that the option idx is in answer_idxs for this batch element
#         mask = torch.zeros((len(self.answer_options)), dtype=torch.bool)
#         mask[answer_idxs] = 1

#         ## Original - NORMAL
#         answer = encode_answer(answer, self.answer_options)
#         tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, self.tokenizer, mode=self.mode, args=self.args)
#         return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
#                torch.tensor(answer, dtype=torch.float), token_type_ids, info, mask
#         ### FLIPPED
#         # flipped_answer = list(set(options)-set(answer))
#         # answer_flipped = encode_answer(flipped_answer, self.answer_options)

#         # tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, self.tokenizer, mode=self.mode, args=self.args)

#         # return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
#         #        torch.tensor(answer_flipped, dtype=torch.float), token_type_ids, info, mask

class RadReStructReversed(Dataset):
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

    def __init__(self, tfm=None, mode='train', args=None):

        self.tfm = tfm
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

        # get file names of data in radrestruct/{split}_qa_pairs
        with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
            self.id_to_img_mapping = json.load(f)

        self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        
        
        ### SINGLE OUT A REPORT
        if mode == 'train':
            self.reports = ['2729.json']
            #self.reports = ['2729_reversed.json']
        
        
        
        
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

        # how many QA pairs
        #self.samples = self.samples[3:4] # For debugging of validation, comment out after
        self.samples = self.samples[0:1] # For debugging of validation, comment out after
        
        for idx, sample in enumerate(self.samples.copy()):
            question = sample[0][0]
            answer = sample[0][1]  # list of correct answers (positive options)
            history = sample[0][2]
            info = sample[0][3]
            ## modifying sample entries
            self.samples[idx][0].append(answer) ## true positive options      
            options = info['options']

            # Create reversed answers (false positives)
            positive_set = set(answer)
            negative_options = [opt for opt in options if opt not in positive_set]

            ### Flip change the positive options at random
            if len(negative_options) >= len(answer):  # only do reversal if we have enough negatives
                false_positives = random.sample(negative_options, len(answer))
                self.samples[idx][0].append(false_positives) # index 4 becomes false positives
            else:
                # fallback if not enough negatives (could skip or handle differently)
                self.samples[idx][0].append([])
            
        #self.samples = self.samples[3:4]

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        qa_sample, img_name, sample_id = self.samples[idx]

        # data_dir/images - c/user/project/data/images
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
        true_positives = qa_sample[4]
        false_positives = qa_sample[5]
        info['positive_option'] = false_positives
        info['true_positive_option'] = true_positives
        answer = false_positives

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

        ## Original - NORMAL
        answer = encode_answer(answer, self.answer_options)
        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, self.tokenizer, mode=self.mode, args=self.args)
        return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
               torch.tensor(answer, dtype=torch.float), token_type_ids, info, mask

class RadReStructCOMBINED(Dataset):
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

    def __init__(self, tfm=None, mode='train', args=None, type='binary'):

        self.tfm = tfm
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

        # get file names of data in radrestruct/{split}_qa_pairs
        with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
            self.id_to_img_mapping = json.load(f)

        self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        
        
        ### SINGLE OUT A REPORT
        if mode == 'train':
            self.reports = ['2729.json']
            #self.reports = ['2729_reversed.json']
        
        
        
        
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


        if type == 'binary':
            self.samples = self.samples[0:1]
            original_sample = self.samples[0]
            
            qa_pair, img, sample_id = original_sample
            question = qa_pair[0]
            true_answers = qa_pair[1]  # the correct answer options (positives)
            history = qa_pair[2]
            info = qa_pair[3]
            
            options = info['options']
            positive_set = set(true_answers)
            new_qa_pair = copy.deepcopy(qa_pair)

            # Find all options that are not in the original true answers
            negative_options = [opt for opt in options if opt not in positive_set]

            # If we have enough negatives, sample a new false set of answers
            if len(negative_options) >= len(true_answers):
                flipped_answers = random.sample(negative_options, len(true_answers))
            else:
                # fallback if not enough negatives
                flipped_answers = []

            # Replace the positive answers with flipped (incorrect) ones
            new_qa_pair[1] = flipped_answers

            # Create new sample with flipped answers
            flipped_sample = (new_qa_pair, img, sample_id + "_flipped")

            # Add to samples
            self.samples.append(flipped_sample)
            print(self.samples)
            
            
        elif type == 'multiclass':
            self.samples = self.samples[3:4] # For debugging of validation, comment out after
            ### 4 C 1 case
            own_sample = []
            own_samples = (['In which part of the body?', 
             ['lung'], 
             [['Are there any foreign objects?', ['yes']], ['Are there foreign bodies?', ['yes']]], 
              {'answer_type':'multi_choice',
                'options': ['abdomen','breast','lung','upper lobe'],
                'path': 'foreign objects_objects_foreign bodies_body_region'}], 'CXR2729_IM-1187-1001', 'CXR2729_IM-1187-1001_2729_3')
        
            own_sample.append(own_samples)
            original_sample = self.samples[0]
            own_ssm = own_sample[0]
            
            qa_pair, img, sample_id = original_sample
            qa_pair_own, img_own, wample_id_own = own_ssm   
            
            question = qa_pair[0]
            true_answers = qa_pair[1]  # the correct answer options (positives)
            history = qa_pair[2]
            info = qa_pair[3]
            
            question_own = qa_pair_own[0]
            true_answers_own = qa_pair_own[1]  # the correct answer options (positives)
            history_own = qa_pair_own[2]
            info_own = qa_pair_own[3]
            
            new_qa_pair = copy.deepcopy(qa_pair)
            
            
            # ### N choose M combinations
            # final_samples = []
            # valid_combination = set()
            # for pos_option_1 in info_own['options']:
            #     for pos_option_2 in info_own['options']:
            #         if pos_option_1 == pos_option_2:
            #             continue
            #         if ((pos_option_1, pos_option_2) not in valid_combination) and ((pos_option_2, pos_option_1) not in valid_combination):
            #             valid_combination.add((pos_option_1,pos_option_2))
            #             valid_options = [pos_option_1, pos_option_2]
            #             new_qa_par_own = copy.deepcopy(qa_pair_own)
            #             new_qa_par_own[1] = valid_options
            #             final_samples.append((new_qa_par_own, img_own, wample_id_own))
                        
            ### 4 choose 1 combinations
            final_samples = []
            valid_combination = set()
            for pos_option_1 in info_own['options']:
                if (pos_option_1 not in valid_combination):
                    valid_combination.add(pos_option_1)
                    valid_options = [pos_option_1]
                    new_qa_par_own = copy.deepcopy(qa_pair_own)
                    new_qa_par_own[1] = valid_options
                    final_samples.append((new_qa_par_own, img_own, wample_id_own))



            self.samples = final_samples


        # Deep copy the sample to modify independently
        
        

                
        
        # ## Scuffed way for binary questions
        # # All possible answer options for this question
        # options = info['options']
        # positive_set = set(true_answers)

        # # Find all options that are not in the original true answers
        # negative_options = [opt for opt in options if opt not in positive_set]

        # # If we have enough negatives, sample a new false set of answers
        # if len(negative_options) >= len(true_answers):
        #     flipped_answers = random.sample(negative_options, len(true_answers))
        # else:
        #     # fallback if not enough negatives
        #     flipped_answers = []

        # # Replace the positive answers with flipped (incorrect) ones
        # new_qa_pair[1] = flipped_answers

        # # Create new sample with flipped answers
        # flipped_sample = (new_qa_pair, img, sample_id + "_flipped")

        # # Add to samples
        # self.samples.append(flipped_sample)
        # print(self.samples)
        

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        qa_sample, img_name, sample_id = self.samples[idx]

        # data_dir/images - c/user/project/data/images
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
        #true_positives = qa_sample[4]
        #false_positives = qa_sample[5]
        info['positive_option'] = answer
        #info['true_positive_option'] = true_positives
        #answer = false_positives

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

        ## Original - NORMAL
        answer = encode_answer(answer, self.answer_options)
        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, self.tokenizer, mode=self.mode, args=self.args)
        return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
               torch.tensor(answer, dtype=torch.float), token_type_ids, info, mask

class RadReStructCOMBINEDEval(Dataset):
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

    def __init__(self, tfm=None, mode='train', args=None, type='binary'):

        self.tfm = tfm
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

        # get file names of data in radrestruct/{split}_qa_pairs
        with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
            self.id_to_img_mapping = json.load(f)

        self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        
        
        ### SINGLE OUT A REPORT
        if mode == 'train':
            self.reports = ['2729.json']
            #self.reports = ['2729_reversed.json']
        
        
        
        
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


        if type == 'binary':
            # how many QA pairs
            self.samples = self.samples[0:1]  # For debugging of validation, comment out after

            original_sample = self.samples[0]
            qa_pair, img, sample_id = original_sample

            question = qa_pair[0]
            true_answers = qa_pair[1]  # the correct answer options (positives)
            history = qa_pair[2]
            info = qa_pair[3]

            # Deep copy the sample to modify independently
            new_qa_pair = copy.deepcopy(qa_pair)
            # reverse the normal order - [yes,no] -> [no, yes]
            new_qa_pair[3]['options'].reverse()
            # All possible answer options for this question
            options = info['options']
            options.reverse()
            positive_set = set(true_answers)

            # Find all options that are not in the original true answers
            negative_options = [opt for opt in options if opt not in positive_set]

            # If we have enough negatives, sample a new false set of answers
            if len(negative_options) >= len(true_answers):
                flipped_answers = random.sample(negative_options, len(true_answers))
            else:
                # fallback if not enough negatives
                flipped_answers = []

            # Replace the positive answers with flipped (incorrect) ones
            new_qa_pair[1] = flipped_answers

            # Create new sample with flipped answers
            flipped_sample = (new_qa_pair, img, sample_id + "_flipped")

            # Add to samples
            self.samples.append(flipped_sample)
            print(self.samples)
            self.samples.reverse()
            print(self.samples)
        elif type == 'multiclass':
            #self.samples = self.samples[3:4] # For debugging of validation, comment out after
            
            self.samples = self.samples[3:4] # For debugging of validation, comment out after
            ### 4 C 1 case
            own_sample = []
            own_samples = (['In which part of the body?', 
             ['lung'], 
             [['Are there any foreign objects?', ['yes']], ['Are there foreign bodies?', ['yes']]], 
              {'answer_type':'multi_choice',
                'options': ['abdomen','breast','lung','upper lobe'],
                'path': 'foreign objects_objects_foreign bodies_body_region'}], 'CXR2729_IM-1187-1001', 'CXR2729_IM-1187-1001_2729_3')
        
            own_sample.append(own_samples)
            original_sample = self.samples[0]
            own_ssm = own_sample[0]
            
            qa_pair, img, sample_id = original_sample
            qa_pair_own, img_own, wample_id_own = own_ssm   
            
            
            question_own = qa_pair_own[0]
            true_answers_own = qa_pair_own[1]  # the correct answer options (positives)
            history_own = qa_pair_own[2]
            info_own = qa_pair_own[3]
            info_own['options'].reverse()
            
            new_qa_pair = copy.deepcopy(qa_pair)
            
            
            # ### N choose M combinations
            # final_samples = []
            # valid_combination = set()
            # for pos_option_1 in info_own['options']:
            #     for pos_option_2 in info_own['options']:
            #         if pos_option_1 == pos_option_2:
            #             continue
            #         if ((pos_option_1, pos_option_2) not in valid_combination) and ((pos_option_2, pos_option_1) not in valid_combination):
            #             valid_combination.add((pos_option_1,pos_option_2))
            #             valid_options = [pos_option_1, pos_option_2]
            #             new_qa_par_own = copy.deepcopy(qa_pair_own)
            #             new_qa_par_own[1] = valid_options
            #             final_samples.append((new_qa_par_own, img_own, wample_id_own))
                        
            ### 4 choose 1 combinations
            final_samples = []
            valid_combination = set()
            for pos_option_1 in info_own['options']:
                if (pos_option_1 not in valid_combination):
                    valid_combination.add(pos_option_1)
                    valid_options = [pos_option_1]
                    new_qa_par_own = copy.deepcopy(qa_pair_own)
                    new_qa_par_own[1] = valid_options
                    final_samples.append((new_qa_par_own, img_own, wample_id_own))



            self.samples = final_samples
            pass




    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, idx):
        qa_sample, img_name, sample_id = self.samples[idx]

        # data_dir/images - c/user/project/data/images
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
        #true_positives = qa_sample[4]
        #false_positives = qa_sample[5]
        info['positive_option'] = answer
        #info['true_positive_option'] = true_positives
        #answer = false_positives

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

        ## Original - NORMAL
        answer = encode_answer(answer, self.answer_options)
        tokens, q_attn_mask, attn_mask, token_type_ids = encode_text_progressive(question, history, self.tokenizer, mode=self.mode, args=self.args)
        return img, torch.tensor(tokens, dtype=torch.long), torch.tensor(q_attn_mask, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long), \
               torch.tensor(answer, dtype=torch.float), token_type_ids, info, mask

class RadReStructEval(Dataset):
    """dataset for report-level inference

        returns:
            img: transformed image tensor
            report: report dict in format of report jsons including labels
            report_vector_gt: ground truth in vectorized form (multi-hot encoding of all selected answers of length 2470)
    """

    def __init__(self, tfm=None, mode='val', args=None, limit_data=None):

        self.tfm = tfm
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

        # get file names of data in radrestruct/{split}_qa_pairs
        with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
            self.id_to_img_mapping = json.load(f)

        self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
        if limit_data is not None:
            self.reports = self.reports[:limit_data]  # for evaluation during training

        # all images corresponding to reports in radrestruct/{split}_qa_pairs
        self.samples = []
        for report_name in self.reports:
            # load report
            with open(f'data/radrestruct/new_reports/{report_name}', 'r') as f:
                report = json.load(f)

            # load target
            with open(f'data/radrestruct/{mode}_vectorized_answers/{report_name}', 'r') as f:
                report_dict = json.load(f)
            report_vector_gt = list(report_dict.values())

            report = {report_name.split('.')[0]: report}
            for elem in self.id_to_img_mapping[report_name.split('.')[0]]:
                self.samples.append((report, elem, report_vector_gt))

        self.mode = mode
        with open('data/radrestruct/answer_options.json', 'r') as f:
            self.answer_options = json.load(f)

        self.samples = self.samples[:1] # Only for debugging, delete afterwards

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        report, img_name, report_vector_gt = self.samples[idx]
        # data_dir/images - c/user/project/data/images
        images_path = os.path.join(self.args.data_dir, 'images')
        img_path = Path(images_path) / f'{img_name}.png'
        img = Image.open(img_path)
        if self.tfm:
            img = self.tfm(img)

        return img, report, torch.tensor(report_vector_gt, dtype=torch.long)


# ### New
# class RadReStructEval(Dataset):
#     """dataset for report-level inference

#         returns:
#             img: transformed image tensor
#             report: report dict in format of report jsons including labels
#             report_vector_gt: ground truth in vectorized form (multi-hot encoding of all selected answers of length 2470)
#     """

#     def __init__(self, tfm=None, mode='val', args=None, limit_data=None):

#         self.tfm = tfm
#         self.args = args

#         self.tokenizer = AutoTokenizer.from_pretrained(args.bert_model, trust_remote_code=True)

#         # get file names of data in radrestruct/{split}_qa_pairs
#         with open("data/radrestruct/id_to_img_mapping_frontal_reports.json", "r") as f:
#             self.id_to_img_mapping = json.load(f)

#         self.reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
#         if limit_data is not None:
#             self.reports = self.reports[:limit_data]  # for evaluation during training


#         ### SINGLE OUT A REPORT
#         #self.reports = ['2729.json']
#         self.reports = self.reports[0:1]


#         # all images corresponding to reports in radrestruct/{split}_qa_pairs
#         self.samples = []
#         for report_name in self.reports:
#             # load report
#             with open(f'data/radrestruct/new_reports/{report_name}', 'r') as f:
#                 report = json.load(f)

#             # load target
#             with open(f'data/radrestruct/{mode}_vectorized_answers/{report_name}', 'r') as f:
#                 report_dict = json.load(f)
#             report_vector_gt = list(report_dict.values())

#             report = {report_name.split('.')[0]: report}
#             for elem in self.id_to_img_mapping[report_name.split('.')[0]]:
#                 self.samples.append((report, elem, report_vector_gt))

#         self.mode = mode
#         with open('data/radrestruct/answer_options.json', 'r') as f:
#             self.answer_options = json.load(f)

#         self.samples = self.samples[4:5] # Only for debugging, delete afterwards

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         report, img_name, report_vector_gt = self.samples[idx]
#         # data_dir/images - c/user/project/data/images
#         images_path = os.path.join(self.args.data_dir, 'images')
#         img_path = Path(images_path) / f'{img_name}.png'
#         img = Image.open(img_path)
#         if self.tfm:
#             img = self.tfm(img)

#         return img, report, torch.tensor(report_vector_gt, dtype=torch.long)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finetune on RadReStruct")

    parser.add_argument('--bert_model', type=str, required=False, default="zzxslp/RadBERT-RoBERTa-4m", help="pretrained question encoder weights")
    parser.add_argument('--max_position_embeddings', type=int, required=False, default=458, help="max length of sequence")
    parser.add_argument('--img_feat_size', type=int, required=False, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=20, help="number of tokens for question")

    args = parser.parse_args()
    # same as vqarad progressive
    args.num_image_tokens = args.img_feat_size ** 2
    args.num_question_tokens = 458 - 3 - args.num_image_tokens

    ds_train = RadReStruct(mode='train', args=args)
    ds_val = RadReStruct(mode='val', args=args)
    ds_test = RadReStruct(mode='test', args=args)

    calculate_bin_weights(ds_train.samples, answer_options=ds_train.answer_options)
