from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Optional, Tuple

from torchvision.transforms import transforms

from PIL import Image

from knowledge_base.constants import Constants, Mode
from knowledge_base.data_utils import load_json_file
from net.image_encoding import ImageEncoderEfficientNet


### To do:
## when retreiving L1 positive paths: lung_signs_yes - make sure to include positives from every l2 positive category
## same when retreiving

class KnowledgeBase:
    def __init__(self, kb_index_file: str):
        self.knowledge_base = load_json_file(kb_index_file)

        self.train_transform = None
        self.test_transform = None
        # self._validate_knowledge_base()

    def get_images_for_path(self, path: str, options_list: list[str]) -> list[Image]:
        required_paths = [f"{path}_{option}" for option in options_list]

        ### Limit to 1 image for now
        images = []
        for option in options_list:
            path = f"{path}_{option}"

            image_path = self.knowledge_base[path][0]
            if len(image_path) > 0:

                image = Image.open(path)
                if self.train_transform:
                    image = self.train_transform(image)

                images.append((option,image))
            else:
                continue


        return images

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