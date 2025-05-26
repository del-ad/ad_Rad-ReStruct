from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_pil_image

from PIL import Image

from knowledge_base.constants import Constants, Mode
from knowledge_base.data_utils import load_json_file
from net.image_encoding import ImageEncoderEfficientNet


### To do:
## when retreiving L1 positive paths: lung_signs_yes - make sure to include positives from every l2 positive category
## same when retreiving

### should only retreive data - given a path - return list of images for that path

class KnowledgeBase:
    def __init__(self, kb_index_file: str):
        self.knowledge_base = load_json_file(Path(kb_index_file))

        self.train_transform = None
        self.test_transform = None
        # self._validate_knowledge_base()

    def get_images_for_path(self, path: str, options_list: List[str], positive_options: List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        positive_paths = [f"{path}_{positive_option}" for positive_option in positive_options]
        negative_paths = [f"{path}_{negative_option}" for negative_option in list(set(options_list)-set(positive_options))]

        ### Limit to 1 image for now
        
        positive_images = self._get_positive_examples(positive_paths)
        negative_images = self._get_negative_examples(negative_paths)
        

        return positive_images, negative_images
    
    # def get_images_for_paths(self, path: str, options_list: List[str]):
    #     examples = self._get_examples(path, options_list)
        
    #     return examples
    
    def get_images_for_paths(self, batch_metadata: List[Dict]):
        batch_examples = []
        for batch_metadata_dict in batch_metadata:
            examples = self._get_examples(batch_metadata_dict['path'], batch_metadata_dict['options'])
            batch_examples.append(examples)
        
        return batch_examples
    
    ## given a positive path - get a positive image
    ## needs to be reworked
    ## for L1 and L2 questions = need avg of positive samples
    ## for L3 - just any positive sample ??
    def _get_positive_examples(self, positive_paths: List[str], num_samples=None) -> List[torch.Tensor]:
        positive_example_images: List[torch.Tensor] = []
        
        
        # import subprocess
        # import os

        # print(f"Running on node: {os.uname().nodename}")

        # print("\nMounted filesystems:")
        # result = subprocess.run(["mount"], capture_output=True, text=True)
        # # Only show actual filesystems
        # print('\n'.join([line for line in result.stdout.splitlines() if line.startswith("/")]))
        
        
        #### need to modify in case several correct options
        for path in positive_paths:
            # list of first 5 images for that path
            image_paths: List[str] = self.knowledge_base[path][0:5]
            images: List[torch.Tensor] = []
            #avg_image = None
            for idx, image_path in enumerate(image_paths):
                print(os.path.exists(Path(image_path)))
                image = Image.open(Path(image_path))
                ## preview the image
                #image.save(f"/home/guests/adrian_delchev/preview_images/pre_transform_positive_{idx}.jpeg")
                if self.train_transform:
                    image = self.train_transform(image)
                #transformed_image = to_pil_image(self.unnormalize(image))
                #transformed_image.save(f"/home/guests/adrian_delchev/preview_images/post_transform_positive_{idx}.jpeg")
                images.append(image)
            #mean_image = self.compute_mean_tensor_image(images)
            #mn_img_as_img = to_pil_image(mean_image)
            #mn_img_as_img.save("/home/guests/adrian_delchev/preview_images/mean_image_unnormalized.jpeg")
            #mean_for_vizualization = self.unnormalize(mean_image)
            #mn_img_as_img = to_pil_image(mean_for_vizualization)
            #mn_img_as_img.save("/home/guests/adrian_delchev/preview_images/mean_image_normalized.jpeg")
            
            #positive_example_images.append(mean_image)
            
                             
        return images
    
    def _get_examples(self, path: str, options_list: List[str], num_samples=None) -> Dict[str,List[torch.Tensor]]:
        example_images: List[torch.Tensor] = []
        
        examples = {}
        
        for option in options_list:
            examples[option] = []
            
            kb_path = f"{path}_{option}"
            ### Need to address
            kb_path = kb_path.replace('body_region', 'body region')
            kb_path = kb_path.replace('body_regions', 'body regions')
            ## get 5 images per path
            image_paths: List[str] = self.knowledge_base[kb_path][0:5]
            images: List[torch.Tensor] = []
            for idx, image_path in enumerate(image_paths):
                #print(os.path.exists(Path(image_path)))
                image = Image.open(Path(image_path))
                ## preview the image
                #image.save(f"/home/guests/adrian_delchev/preview_images/pre_transform_positive_{idx}.jpeg")
                if self.train_transform:
                    image = self.train_transform(image)
                #transformed_image = to_pil_image(self.unnormalize(image))
                #transformed_image.save(f"/home/guests/adrian_delchev/preview_images/post_transform_positive_{idx}.jpeg")
                images.append(image)
            examples[option] = images
                
                             
        return examples
                
    def _get_negative_examples(self, negative_paths: list[str], num_samples=None) -> List[torch.Tensor]:
        negative_example_images = []
        
        
        ### need to modify in case of several incorrect paths
        for path in negative_paths:
            # list of first 5 images for that path
            image_paths: List[str] = self.knowledge_base[path][0:5]
            images: List[torch.Tensor] = []
            avg_image = None
            for image_path in image_paths:
                image = Image.open(image_path)
                if self.train_transform:
                    image = self.train_transform(image)
                images.append(image)
            #mean_image = self.compute_mean_tensor_image(images)
                
            
            
            #negative_example_images.append(mean_image)
            
        return images


    def unnormalize(self, img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (img_tensor * std) + mean
    def compute_mean_tensor_image(self,tensor_list: list[torch.Tensor]) -> torch.Tensor:
        # tensor_list contains CxHxW tensors, e.g., from transforms.ToTensor()
        
        # Stack into shape (N, C, H, W)
        stacked = torch.stack(tensor_list, dim=0)
        
        # Compute mean across the batch (dim=0)
        mean_tensor = torch.mean(stacked, dim=0)
        
        return mean_tensor  # Still a CxHxW tensor
                

    def _validate_image(self, image_path: str) -> Tuple[bool, Optional[str]]:
        if not os.path.isfile(image_path):
            return False, f"Path is not a file: {image_path}"
        try:
            with Image.open(image_path) as img:
                img.verify()  # Ensure image is not corrupted
            return True, None
        except Exception as e:
            return False, f"Cannot load image: {image_path} - {e}"

    def _validate_knowledge_base(self):
        start_time = time.time()

        all_image_paths = [
            image_path
            for images_obj_list in self.knowledge_base.values()
            for image_path in images_obj_list
        ]

        print(f"Validating {len(all_image_paths)} images...")

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(self._validate_image, all_image_paths))

        failed = [error for ok, error in results if not ok]
        if failed:
            raise Exception(f"Validation failed for {len(failed)} images:\n" + "\n".join(failed))

        elapsed_time = time.time() - start_time
        print(f"✅ Knowledge base validation passed ({len(all_image_paths)} images checked)")
        print(f"⏱️ Validation completed in {elapsed_time:.2f} seconds")


## should convert img -> image embedding, text -> text embedding
class KnowledgeBasePostProcessor:
    def __init__(self, text_encoder, image_encoder) -> None:
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        
    def encode_text(self, positive_answers: List[str], negative_answers: List[str]):
        # attn_mask = (args.num_image_tokens + len(tokens)) * [1] + (args.max_position_embeddings - len(tokens) - args.num_image_tokens) * [0]
        pos_answers = []
        neg_answers = []
        for positive_answer in positive_answers:
            # text = f":{positive_answer}"
            # input_ids = self.text_encoder.tokenizer.encode(text)
            # input_ids_nofl = self.text_encoder.tokenizer.encode(text)[1:-1]
            # # for debugging
            # decoded_text = self.text_encoder.tokenizer.decode(input_ids)
            # decoded_text_nofl = self.text_encoder.tokenizer.decode(input_ids_nofl)
            
            # n_pad = self.text_encoder.args.max_position_embeddings - len(input_ids)
            # n_pad_nofl = self.text_encoder.args.max_position_embeddings - len(input_ids_nofl)
            
            # #n_pad = args.max_position_embeddings - len(tokens)
            # #tokens.extend([tokenizer.pad_token_id] * n_pad)
            
            # attn_mask = len(input_ids) * [1] + (self.text_encoder.args.max_position_embeddings - len(input_ids)) * [0]
            # attn_mask_nofl = len(input_ids_nofl) * [1] + (self.text_encoder.args.max_position_embeddings - len(input_ids_nofl)) * [0]
            
            # input_ids.extend([self.text_encoder.tokenizer.pad_token_id] * n_pad)
            # input_ids_nofl.extend([self.text_encoder.tokenizer.pad_token_id] * n_pad_nofl)
            
            # text_embeddings = self.text_encoder(self.prepare_list_for_model(input_ids, self.text_encoder), self.prepare_list_for_model(attn_mask, self.text_encoder))
            # text_embeddings_nofl = self.text_encoder(self.prepare_list_for_model(input_ids_nofl, self.text_encoder), self.prepare_list_for_model(attn_mask_nofl, self.text_encoder))
            
            self.text_encoder.eval()

            
            tokenized = self.text_encoder.tokenizer(
                positive_answer,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device=next(self.text_encoder.parameters()).device)
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            with torch.no_grad():
                pos_answer_embedding = self.text_encoder(input_ids, attention_mask)
                pos_answers.append(pos_answer_embedding)
            
            # inputs = self.text_encoder.tokenizer(inputs, return_tensor='pt', add_special_tokens=False)
            # outputs = self.text_encoder(**inputs)
            
        for positive_answer in positive_answers:
            self.text_encoder.eval()
            tokenized = self.text_encoder.tokenizer(
                positive_answer,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device=next(self.text_encoder.parameters()).device)
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            
            with torch.no_grad():
                neg_answer_embedding = self.text_encoder(input_ids, attention_mask)
                neg_answers.append(neg_answer_embedding)
            
            # inputs = self.text_encoder.tokenizer(inputs, return_tensor='pt', add_special_tokens=False)
            # outputs = self.text_encoder(**inputs)
        
        ## embedding of column
        tokenized_column = self.text_encoder.tokenizer(
            ":",
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device=next(self.text_encoder.parameters()).device)
        input_ids = tokenized_column['input_ids']
        attention_mask = tokenized_column['attention_mask']
        
        with torch.no_grad():
            column_embedding = self.text_encoder(input_ids, attention_mask)
        
        #model_ready_answer
        return pos_answers, neg_answers, column_embedding
    
    
    def encode_text_batch(self, positive_answers: List[str], negative_answers: List[str]):
        options_embeddings = {}
        positive_ans_embeddings = self.text_encoder.encode_phrases(positive_answers)
        neg_ans_embeddings = self.text_encoder.encode_phrases(negative_answers)
        column_embedding = self.__get_column_embedding()
        sep_embedding = self.__get_sep_embedding()
        
        return positive_ans_embeddings, neg_ans_embeddings, column_embedding, sep_embedding
    
    def encode_options(self, options: List[str]) -> Tuple[Dict[str, List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        options_embeddings = {}
        for option in options:
            options_embeddings[option] = self.__get_embedding(option)
        
        #options_embeddings = self.text_encoder.encode_options(options)
        column_embedding = self.__get_column_embedding()
        sep_embedding = self.__get_sep_embedding()
        
        return options_embeddings, column_embedding, sep_embedding
    
    def encode_batch_options(self, batch_metadata: List[Dict]) -> Tuple[List[Dict[str, List[torch.Tensor]]], torch.Tensor, torch.Tensor]:
        
        batch_options_embeddings = []
        for batch in batch_metadata:
            options = batch['options']
            options_embeddings = {}
            for option in options:
                options_embeddings[option] = self.__get_embedding(option)
            batch_options_embeddings.append(options_embeddings)
            
        
        #options_embeddings = self.text_encoder.encode_options(options)
        column_embedding = self.__get_column_embedding()
        sep_embedding = self.__get_sep_embedding()
        
        return batch_options_embeddings, column_embedding, sep_embedding
    
    def __get_column_embedding(self):
        colon_token = ":"
        tokenized = self.text_encoder.tokenizer(colon_token, add_special_tokens=False)
        colon_token_id = tokenized["input_ids"][0]
        
        text_encoder_embeddings_matrix = self.text_encoder.BERTmodel.get_input_embeddings()
        colon_embedding = text_encoder_embeddings_matrix(torch.tensor(colon_token_id)
                                                         .to(device=next(self.text_encoder.parameters()).device))
        return colon_embedding.unsqueeze(0).unsqueeze(0)
        
    def __get_sep_embedding(self):
        text_encoder_embeddings_matrix = self.text_encoder.BERTmodel.get_input_embeddings()
        sep_token_embedding = text_encoder_embeddings_matrix(torch.tensor(self.text_encoder.tokenizer.sep_token_id)
                                                         .to(device=next(self.text_encoder.parameters()).device))
        return sep_token_embedding.unsqueeze(0).unsqueeze(0)
    
    def __get_embedding(self, option: str):
        # text_encoder_embeddings_matrix = self.text_encoder.BERTmodel.get_input_embeddings()
        # device = next(self.text_encoder.parameters()).device

        # words = option.strip().split()

        # assert len(words) > 0  # Catch empty strings

        # Single word option
        # if len(words) == 1:
        #     tokenized = self.text_encoder.tokenizer(option, add_special_tokens=False)
        #     token_ids = tokenized["input_ids"]

        #     # Handle subwords by averaging their embeddings
        #     token_embeddings = [
        #         text_encoder_embeddings_matrix(torch.tensor(token_id).to(device))
        #         for token_id in token_ids
        #     ]
        #     token_embedding = torch.stack(token_embeddings, dim=0).mean(dim=0)
        #     return [token_embedding.unsqueeze(0).unsqueeze(0)]

        # else:
        #     word_embeddings = []
        #     for word in words:
        #         tokenized = self.text_encoder.tokenizer(word, add_special_tokens=False)
        #         token_ids = tokenized["input_ids"]

        #         token_embeddings = [
        #             text_encoder_embeddings_matrix(torch.tensor(token_id).to(device))
        #             for token_id in token_ids
        #         ]
        #         word_embedding = torch.stack(token_embeddings, dim=0).mean(dim=0)
        #         word_embeddings.append(word_embedding.unsqueeze(0).unsqueeze(0))

        #     #mean_option_embedding = torch.stack(word_embeddings, dim=0).mean(dim=0)
        #     return word_embeddings
        
            # Get the embedding layer (vocab_size x embedding_dim)
        text_encoder_embeddings_matrix = self.text_encoder.BERTmodel.get_input_embeddings()
        device = next(self.text_encoder.parameters()).device

        # Tokenize without adding special tokens (e.g., [CLS], [SEP])
        tokenized = self.text_encoder.tokenizer(option, add_special_tokens=False, return_tensors="pt")
        
        # Extract token IDs and move them to the correct device
        input_ids = tokenized["input_ids"].to(device)  # Shape: (1, num_tokens)
        decoded = self.text_encoder.tokenizer.decode(input_ids[0])
        # Pass through embedding layer: output shape (1, num_tokens, embedding_dim)
        token_embeddings = text_encoder_embeddings_matrix(input_ids)

        # Reshape to (num_tokens, 1, embedding_dim)
        token_embeddings = token_embeddings.squeeze(0).unsqueeze(1)  # (num_tokens, 1, embedding_dim)

        
        return token_embeddings
                
        
    
    ## given an image - > get its image features and global embedding
    def encode_image(self, image: torch.Tensor):
        ## just move image to gpu, set to correct tensor format for model input
        model_ready_image = self.prepare_tensor_for_model(image, self.image_encoder)
        return self.image_encoder(model_ready_image, mode='train')
    
    def prepare_tensor_for_model(self, tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        Ensures tensor has a batch dimension and matches the model's device and dtype.
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension if needed
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return tensor.to(device=device, dtype=dtype)
    
    def prepare_list_for_model(self, list: list[int], model: torch.nn.Module) -> torch.Tensor:
        """
        Ensures tensor has a batch dimension and matches the model's device and dtype.
        """

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        tensor = torch.tensor(list, dtype=torch.long, device=device).unsqueeze(0)
        return tensor.to(device=device, dtype=torch.long)
    

if __name__ == '__main__':
    CONSTANTS = Constants(Mode.CLUSTER)

    KNOWLEDGE_BASE = KnowledgeBase(CONSTANTS.KNOWLEDGE_BASE_INDEX_FILE)
    images = KNOWLEDGE_BASE.get_images_for_path('lung_signs_yes')
    print("done :)")