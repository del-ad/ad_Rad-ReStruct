import argparse
import json
import cv2
from torchvision.transforms.functional import to_pil_image
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
        special_paths = {'lung_body_regions_localization', 'lung_body_regions_attributes', 'lung_body_regions_degree', 
                         'trachea_body_regions_attributes', 'trachea_body_regions_degree',
                         'pleura_body_regions_localization', 'pleura_body_regions_attributes','pleura_body_regions_degree'}
        
        examples = {}
        
        for option in options_list:
            examples[option] = []
            
            kb_path = f"{path}_{option}"
            ### Need to address
            ### happens very rarely
            if path in special_paths:
                path_elements = path.split("_")
                last = path_elements[-1]
                without_last = path_elements[0:-1]
                without_last.append(path_elements[0])
                without_last.append(last)
                
                #path_elements.append(path_elements[0])
                kb_path = f"{'_'.join(without_last)}_{option}"
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

class CachedKnowledgeBase:
    def __init__(self, knowledge_base, full_transform, img_transform, norm_transform):
        self.image_cache = {}
        self.transform = full_transform
        self.img_transform = img_transform
        self.norm_transform = norm_transform
        self.kb = load_json_file(Path(knowledge_base))
        self.labels_encodings = {}
        
        self.special_paths_l3 = {'lung_body_regions_localization', 'lung_body_regions_attributes', 'lung_body_regions_degree', 
                         'trachea_body_regions_attributes', 'trachea_body_regions_degree',
                         'pleura_body_regions_localization', 'pleura_body_regions_attributes','pleura_body_regions_degree',}
        
        self.special_paths_l1 = {'lung_body_regions','lung_body_regions','trachea_body_regions','trachea_body_regions','pleura_body_regions','pleura_body_regions'}
        
        self.path_lookup = self.__generate_path_lookup()
        
        # with open(f'data/radrestruct/path_lookup.json', 'r') as f:
        #     self.path_lookup = json.load(f)
        
        ### Load in full size, then resize
        # print("Preloading knowledge base...")
        # for idx, (kb_key, image_paths) in enumerate(self.kb.items()):
        #     self.image_cache[kb_key] = []
        #     for img_path in image_paths[:5]:
        #         image = Image.open(img_path)
        #         image_tensor = self.transform(image)  # e.g., Resize + ToTensor + Normalize
        #         self.image_cache[kb_key].append(image_tensor)
        #     print(f"Preload complete for {kb_key} | {idx}/{len(self.kb)}")
        
        ## Load in full size, then resize
        print("Preloading knowledge base...")
        for idx, (kb_key, image_paths) in enumerate(self.kb.items()):
            if idx >= 0:
                self.image_cache[kb_key] = []
                for kb_img_path in image_paths[:5]:
                    ####### Load using OpenCV - load, resize, transform
                    #img = self.load_kb_image_cv2(kb_img_path, self.img_transform, self.norm_transform)
                    #self.image_cache[kb_key].append(img)
                    
                    ####### Load ready images using PIL
                    img = self.load_kb_image_PIL(kb_img_path, self.img_transform, self.norm_transform)
                    self.image_cache[kb_key].append(img)
                    
                    ####### ---- SAVE IMAGES INSTEAD
                    #self.save_kb_images_cv2(kb_img_path, self.img_transform, self.norm_transform)
                    
                #self.image_cache[kb_key] = [self.load_kb_image_cv2(img_path, self.img_transform, self.norm_transform) for img_path in image_paths[:5]]
                print(f"Preload complete for {kb_key} | {idx}/{len(self.kb)}")
            
        print(f"NUM OF IMAGES: {sum((len(value) for key,value in self.image_cache.items()))}")
            
    # def get_images_for_paths(self, batch_metadata):
    #     batch_examples = []
    #     for meta in batch_metadata:
    #         path = meta["path"]
    #         options = meta["options"]
    #         examples = {}
    #         for option in options:
    #             key = f"{path}_{option}"  # Apply same rules as your special handling
    #             if path in self.special_paths:
    #                 path_elements = path.split("_")
    #                 last = path_elements[-1]
    #                 without_last = path_elements[0:-1]
    #                 without_last.append(path_elements[0])
    #                 without_last.append(last)
                    
    #                 #path_elements.append(path_elements[0])
    #                 kb_path = f"{'_'.join(without_last)}_{option}"
    #             kb_path = key.replace('body_region', 'body region')
    #             kb_path = kb_path.replace('body_regions', 'body regions')
                
                
    #             examples[option] = self.image_cache.get(kb_path, [])
    #         batch_examples.append(examples)
    #     return batch_examples
    
    
    ### given a path - return the knowledge base examples for 
    ## that path - 
    def get_images_for_paths(self, batch_metadata):
        batch_examples = []
        for meta in batch_metadata:
            path = meta["path"]
            options = meta["options"]
            examples = {}
            for option in options:
                lookup_tuple = (path,option)
                
                examples[option] = self.image_cache.get(self.path_lookup[lookup_tuple], [])
            batch_examples.append(examples)
        return batch_examples
    
    def __generate_path_lookup(self,):
        path_dict = {}
        for mode in ['train', 'val', 'test']:
            reports = sorted(os.listdir(f'data/radrestruct/{mode}_qa_pairs'))
            # all images corresponding to reports in radrestruct/{split}_qa_pairs
            for report in reports:
                with open(f'data/radrestruct/{mode}_qa_pairs/{report}', 'r') as f:
                    qa_pairs = json.load(f)
                    
                    for qa_pair in qa_pairs:
                        info = qa_pair[3]
                        base_path = info['path']
                        path_options = info['options']
                        for option in path_options:
                            path = f"{base_path}_{option}"
                            kb_path = f"{base_path}_{option}"
                            
                            ### need to add lung/pleura/trachea ->lung_body_regions ->lung_body_regions_lung
                            new_base = None
                            if base_path in self.special_paths_l3:
                                path_elements = base_path.split("_")
                                last = path_elements[-1]
                                without_last = path_elements[0:-1]
                                without_last.append(path_elements[0])
                                without_last.append(last)
                                new_base = str("_".join(without_last))

                                # if last != 'yes' and last != 'no':
                                #     without_last.append(last)
                            
                            if ((base_path in self.special_paths_l1) and (option in {'yes','no'})):
                                path_elements = base_path.split("_")
                                path_elements.append(path_elements[0])
                                #without_last = path_elements
                                new_base = str("_".join(path_elements))


                        
                        #path_elements.append(path_elements[0])
                                kb_path = f"{new_base}_{option}"
                                #kb_orig = str(kb_orig)

                            kb_path = kb_path.replace('body_region', 'body region')
                            kb_path = kb_path.replace('body_regions', 'body regions')
                            path_dict[(base_path,option)] = kb_path
                            if 'body_regions' in base_path or 'body_region' in base_path:
                                base_path_br = str(base_path)
                                base_path_br = base_path_br.replace('body_region', 'body region')
                                base_path_br = base_path_br.replace('body_regions', 'body regions')
                                path_dict[(base_path_br,option)] = kb_path
                            if new_base:
                                path_dict[(new_base,option)] = kb_path
                                new_base = new_base.replace('body_region', 'body region')
                                new_base = new_base.replace('body_regions', 'body regions')
                                path_dict[(new_base,option)] = kb_path

                            # kb_path = kb_path.replace('body_region', 'body region')
                            # kb_path = kb_path.replace('body_regions', 'body regions')
                            #unique_paths.add(path)

        
        
        
        return path_dict
    

    def load_kb_image_cv2(self, image_path, img_tfm, norm_tfm, resize_size=(488, 488)):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        split = image_path.split("/")
        last = split[-1]
        file_name = last.split(".")[0]

        if img is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)
        img = np.stack([img] * 3, axis=-1)  # HWC, 3-channel

        # Convert NumPy array (HWC, uint8) to PIL Image
        img = Image.fromarray(img)
        
        pil_img = to_pil_image(img)
        img.save(f"/home/guests/adrian_delchev/code/kb_images/{file_name}.png")

        # Now apply img_tfm (e.g., Resize/Crop etc.)
        #img = img_tfm(img)

        # If img_tfm includes ToTensor, then just:
        #img = norm_tfm(img)
        
        # pil_img = to_pil_image(img)
        # nd_img = np.array(img)
        # print(f"Image shape: {nd_img.shape}")
        # print(f"Bytes in memory: {nd_img.nbytes} bytes")
        # print(f"Kilobytes: {nd_img.nbytes / 1024:.2f} KB")
        # print(f"Megabytes: {nd_img.nbytes / 1024**2:.2f} MB")
        
        # print("PYTORCH TENSOR:")
        # n_bytes = img.numel() * img.element_size()
        # print(f"Image shape: {img.shape}")
        # print(f"Bytes in memory: {n_bytes}")
        # print(f"Kilobytes: {n_bytes / 1024:.2f} KB")
        # print(f"Megabytes: {n_bytes / (1024*1024):.2f} MB")
        # pil_img.save(f"/home/guests/adrian_delchev/code/kb_images/{file_name}.png")
        # pil_img.save(f"/home/guests/adrian_delchev/code/kb_images/{file_name}.jpeg")
        # # self.save_normalized_tensor_image(img, self.norm_transform, "img_1.jpeg" ) or png
        img.requires_grad = False
        return img
    
    def load_kb_image_PIL(self, image_path, img_tfm, norm_tfm, resize_size=(488, 488)):
        img = Image.open(image_path)
        

        if img is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")


        img = img_tfm(img)
        # If img_tfm includes ToTensor, then just:
        img = norm_tfm(img)
        
        img.requires_grad = False
        return img
    
    def save_kb_images_cv2(self, image_path, img_tfm, norm_tfm, resize_size=(488, 488)):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        split = image_path.split("/")
        last = split[-1]
        file_name = last.split(".")[0]

        if img is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        img = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)
        img = np.stack([img] * 3, axis=-1)  # HWC, 3-channel

        # Convert NumPy array (HWC, uint8) to PIL Image
        img = Image.fromarray(img)
        
        img.save(f"/home/guests/adrian_delchev/code/kb_images/{file_name}.png")
        del img

        # Now apply img_tfm (e.g., Resize/Crop etc.)
        #img = img_tfm(img)

        # If img_tfm includes ToTensor, then just:
        #img = norm_tfm(img)
        
        # pil_img = to_pil_image(img)
        # nd_img = np.array(img)
        # print(f"Image shape: {nd_img.shape}")
        # print(f"Bytes in memory: {nd_img.nbytes} bytes")
        # print(f"Kilobytes: {nd_img.nbytes / 1024:.2f} KB")
        # print(f"Megabytes: {nd_img.nbytes / 1024**2:.2f} MB")
        
        # print("PYTORCH TENSOR:")
        # n_bytes = img.numel() * img.element_size()
        # print(f"Image shape: {img.shape}")
        # print(f"Bytes in memory: {n_bytes}")
        # print(f"Kilobytes: {n_bytes / 1024:.2f} KB")
        # print(f"Megabytes: {n_bytes / (1024*1024):.2f} MB")
        # pil_img.save(f"/home/guests/adrian_delchev/code/kb_images/{file_name}.png")
        # pil_img.save(f"/home/guests/adrian_delchev/code/kb_images/{file_name}.jpeg")
        # # self.save_normalized_tensor_image(img, self.norm_transform, "img_1.jpeg" ) or png
        
        return None
    
    def save_normalized_tensor_image(self, tensor_img, norm_transform, save_path):
        """
        Unnormalize and save an image tensor that was normalized using torchvision.transforms.Normalize.

        Args:
            tensor_img (torch.Tensor): Tensor image [C, H, W], normalized.
            norm_transform (transforms.Normalize): The transform used to normalize the image.
            save_path (str): Path to save the output image.
        """
        # Step 1: Unnormalize (reverse Normalize)
        mean = torch.tensor(norm_transform.mean).view(-1, 1, 1)
        std = torch.tensor(norm_transform.std).view(-1, 1, 1)

        unnorm_img = tensor_img.clone() * std + mean  # element-wise reverse

        # Step 2: Clamp to [0, 1] to ensure valid image range
        unnorm_img = torch.clamp(unnorm_img, 0.0, 1.0)

        # Step 3: Convert to PIL image and save
        pil_img = to_pil_image(unnorm_img)  # converts [C, H, W] float tensor to PIL
        pil_img.save(save_path)
## should convert img -> image embedding, text -> text embedding
class KnowledgeBasePostProcessor:
    def __init__(self, text_encoder, image_encoder) -> None:
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.column_embedding = self.__get_column_embedding
        self.sep_embedding = self.__get_sep_embedding
        self.label_embeddings = {}
        self.label_embeddings = self.__generate_options_embeddings()
        
    def encode_text(self, positive_answers: List[str], negative_answers: List[str]):
        # attn_mask = (args.num_image_tokens + len(tokens)) * [1] + (args.max_position_embeddings - len(tokens) - args.num_image_tokens) * [0]
        pos_answers = []
        neg_answers = []
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
    
    def __generate_options_embeddings(self):
        options = load_json_file(Path("/home/guests/adrian_delchev/code/ad_Rad-ReStruct/data/radrestruct/answer_options.json"))
        
        for option in options:
            self.label_embeddings[option] = self.__get_embedding(option)
    
    def encode_options(self, options: List[str]) -> Tuple[Dict[str, List[torch.Tensor]], torch.Tensor, torch.Tensor]:
        options_embeddings = {}
        for option in options:
            options_embeddings[option] = self.__get_embedding(option)
        
        #options_embeddings = self.text_encoder.encode_options(options)
        column_embedding = self.__get_column_embedding()
        sep_embedding = self.__get_sep_embedding()
        
        return options_embeddings, column_embedding, sep_embedding
    
    ### given a batch , produce the option -> embedding dict
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
    
    def encode_batch_options_batched(self, batch_metadata):
        # Step 1: Collect all unique options
        all_options_set = set()
        for batch in batch_metadata:
            all_options_set.update(batch['options'])
        all_options = list(all_options_set)

        # Step 2: Batch tokenize
        encoded = self.text_encoder.tokenizer(
            all_options,
            add_special_tokens=False,
            return_tensors='pt',
            padding=True
        )

        device = next(self.text_encoder.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        embeddings_layer = self.text_encoder.BERTmodel.get_input_embeddings()
        option_embeddings = embeddings_layer(input_ids)  # (num_options, max_len, emb_dim)

        # Step 3: Map each option to its embedding sequence (unpadded)
        option_to_embedding = {}
        for i, option in enumerate(all_options):
            length = encoded['attention_mask'][i].sum().item()
            emb = option_embeddings[i, :length]  # (num_tokens, emb_dim)
            emb = emb.unsqueeze(1)  # (num_tokens, 1, emb_dim)
            option_to_embedding[option] = emb.detach()  # .detach() if needed

        # Step 4: Assemble per-batch dicts
        batch_options_embeddings = []
        for batch in batch_metadata:
            options = batch['options']
            options_embeddings = {option: option_to_embedding[option] for option in options}
            batch_options_embeddings.append(options_embeddings)
            #options_embeddings.clear()

        # Example: handle col_embedding/sep_embedding as before, not batched here
        column_embedding = self.__get_column_embedding()  # recompute per forward
        sep_embedding = self.__get_sep_embedding()
        
        option_to_embedding.clear()
        del option_to_embedding

        return batch_options_embeddings, column_embedding, sep_embedding
    
    ### given a batch , produce the option -> embedding dict
    def encode_batch_options_fast(self, batch_metadata: List[Dict]) -> Tuple[List[Dict[str, List[torch.Tensor]]], torch.Tensor, torch.Tensor]:
        
        
        batch_options_embeddings = []
        for batch in batch_metadata:
            options = batch['options']
            # for option in options:
            #     options_embeddings[option] = self.label_embeddings[option]
            batch_options_embeddings.append({option: self.label_embeddings[option] for option in options})
            
        
        #options_embeddings = self.text_encoder.encode_options(options)
        column_embedding = self.column_embedding()
        sep_embedding = self.sep_embedding()
        
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
    parser = argparse.ArgumentParser(description="Test knowledge base")

    parser.add_argument('--run_name', type=str, required=False, default="debug", help="run name for wandb")
    parser.add_argument('--data_dir', type=str, required=False, default="data/radrestruct", help="path for data")
    parser.add_argument('--model_dir', type=str, required=False, default="", help="path to load weights")
    parser.add_argument('--save_dir', type=str, required=False, default="checkpoints_radrestruct", help="path to save weights")
    parser.add_argument('--question_type', type=str, required=False, default=None, help="choose specific category if you want")
    parser.add_argument('--use_pretrained', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="use mixed precision or not")
    parser.add_argument('--bert_model', type=str, required=False, default="zzxslp/RadBERT-RoBERTa-4m", help="pretrained question encoder weights")

    parser.add_argument('--progressive', action='store_true', default=False, help="use progressive answering of questions")
    parser.add_argument('--match_instances', action='store_true', default=False, help="do optimal instance matching")
    parser.add_argument('--aug_history', action='store_true', default=False, help="do history augmentation")

    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--num_workers', type=int, required=False, default=12, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")
    parser.add_argument('--classifier_dropout', type=float, required=False, default=0.0, help="how often should image be dropped")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=12, help="max length of sequence")
    parser.add_argument('--max_answer_len', type=int, required=False, default=29, help="padding length for free-text answers")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="batch size")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")

    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3, help="hidden dropout probability")

    parser.add_argument('--img_feat_size', type=int, required=False, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=20, help="number of tokens for question")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=16, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="num of fusion layers")
    parser.add_argument('--acc_grad_batches', type=int, required=False, default=None, help="how many batches to accumulate gradients")
    ## KB
    parser.add_argument('--kb_dir', type=str, required=False, default=None, help="the path to the knowledge base index file")
    
    args = parser.parse_args()
    args.num_image_tokens = args.img_feat_size ** 2
    args.max_position_embeddings = 458
    args.hidden_size_img_enc = args.hidden_size
    args.num_question_tokens = 458 - 3 - args.num_image_tokens
    
    image_encoder = ImageEncoderEfficientNet(args)
    
    img_tfm = image_encoder.img_tfm
    norm_tfm = image_encoder.norm_tfm
    resize_size = image_encoder.resize_size
    
    test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm
    
    kb_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                        *img_tfm.transforms])
    
    ## apply the normalization transforms from efficient net
    kb_transforms = transforms.Compose([kb_transforms,
                                        norm_tfm])
    
    
    CONSTANTS = Constants(Mode.CLUSTER)

    KNOWLEDGE_BASE = KnowledgeBase(CONSTANTS.KNOWLEDGE_BASE_INDEX_FILE)
    KNOWLEDGE_BASE.train_transform = kb_transforms
    KNOWLEDGE_BASE_CACHED = CachedKnowledgeBase(CONSTANTS.KNOWLEDGE_BASE_INDEX_FILE, kb_transforms)
    
    sample_paths = [{'path': 'lung_signs',
                    'options': ['yes', 'no']}]
    
    images = KNOWLEDGE_BASE.get_images_for_paths(sample_paths)
    images2 = KNOWLEDGE_BASE_CACHED.get_images_for_paths(sample_paths)
    print("done :)")