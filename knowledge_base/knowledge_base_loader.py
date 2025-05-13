from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import time
from typing import Optional, Tuple, List

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

class KnowledgeBase:
    def __init__(self, kb_index_file: str):
        self.knowledge_base = load_json_file(Path(kb_index_file))

        self.train_transform = None
        self.test_transform = None
        # self._validate_knowledge_base()

    def get_images_for_path(self, path: str, options_list: list[str], positive_options: list[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        positive_paths = [f"{path}_{positive_option}" for positive_option in positive_options]
        negative_paths = [f"{path}_{negative_option}" for negative_option in list(set(options_list)-set(positive_options))]

        ### Limit to 1 image for now
        
        positive_images = self._get_positive_examples(positive_paths)
        negative_images = self._get_negative_examples(negative_paths)
        

        return positive_images, negative_images
    
    ## given a positive path - get a positive image
    ## needs to be reworked
    ## for L1 and L2 questions = need avg of positive samples
    ## for L3 - just any positive sample ??
    def _get_positive_examples(self, positive_paths: List[str], num_samples=None) -> List[torch.Tensor]:
        positive_example_images: List[torch.Tensor] = []
        
        # import subprocess
        # print(f"Running on node: {os.uname().nodename}")
        # print("Mount info for /home:")
        # subprocess.run(["df", "-h", "/home"])
        # print("\nSymlink status of /home/data:")
        # subprocess.run(["ls", "-l", "/home/data"])
        # print("\nDoes the target file exist?")
        
        import subprocess
        import os

        print(f"Running on node: {os.uname().nodename}")

        print("\nMounted filesystems:")
        result = subprocess.run(["mount"], capture_output=True, text=True)
        # Only show actual filesystems
        print('\n'.join([line for line in result.stdout.splitlines() if line.startswith("/")]))
        
        # for path in positive_paths:
        #     # list of first 5 images for that path
        #     image_paths: List[str] = self.knowledge_base[path][0:5]
        #     images: List[torch.Tensor] = []
        #     #avg_image = None
        #     for idx, image_path in enumerate(image_paths):
        #         print(os.path.exists(Path(image_path)))
        #         #image = Image.open(Path(image_path))
        #         ## preview the image
        #         #image.save(f"/home/guests/adrian_delchev/preview_images/pre_transform_positive_{idx}.jpeg")
        #         if self.train_transform:
        #             image = self.train_transform(image)
        #         #transformed_image = to_pil_image(self.unnormalize(image))
        #         #transformed_image.save(f"/home/guests/adrian_delchev/preview_images/post_transform_positive_{idx}.jpeg")
        #         images.append(image)
        #     mean_image = self.compute_mean_tensor_image(images)
        #     #mn_img_as_img = to_pil_image(mean_image)
        #     #mn_img_as_img.save("/home/guests/adrian_delchev/preview_images/mean_image_unnormalized.jpeg")
        #     #mean_for_vizualization = self.unnormalize(mean_image)
        #     #mn_img_as_img = to_pil_image(mean_for_vizualization)
        #     #mn_img_as_img.save("/home/guests/adrian_delchev/preview_images/mean_image_normalized.jpeg")
            
        #     positive_example_images.append(mean_image)
            
                             
        return positive_example_images
                
    def _get_negative_examples(self, negative_paths: list[str], num_samples=None) -> List[torch.Tensor]:
        negative_example_images = []
        
        # for path in negative_paths:
        #     # list of first 5 images for that path
        #     image_paths: List[str] = self.knowledge_base[path][0:5]
        #     images: List[torch.Tensor] = []
        #     avg_image = None
        #     for image_path in image_paths:
        #         image = Image.open(image_path)
        #         if self.train_transform:
        #             image = self.train_transform(image)
        #         images.append(image)
        #     mean_image = self.compute_mean_tensor_image(images)
                
            
            
        #     negative_example_images.append(mean_image)
            
        return negative_example_images


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
                

    def prepare_tensor_for_model(self, tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """
        Ensures tensor has a batch dimension and matches the model's device and dtype.
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension if needed
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return tensor.to(device=device, dtype=dtype)

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



if __name__ == '__main__':
    CONSTANTS = Constants(Mode.CLUSTER)

    KNOWLEDGE_BASE = KnowledgeBase(CONSTANTS.KNOWLEDGE_BASE_INDEX_FILE)
    images = KNOWLEDGE_BASE.get_images_for_path('lung_signs_yes')
    print("done :)")