import os
import json
from tqdm import tqdm
import cv2
import numpy as np
from make_dataset_sample import create_image_with_random_cards
from make_dataset_util import COORD_K, BBOX_K
from constants import CARDS_PATH, BG_PATH, OUTPUT_DFINE_PATH, DFINE_ANNOTATION_OUTPUT

def convert_np_types(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def create_dfine_dataset(n_samples, mode):
    bg_folder = BG_PATH
    cards_path = CARDS_PATH
    output_path = OUTPUT_DFINE_PATH+"\\"+mode
    annotation_path = DFINE_ANNOTATION_OUTPUT

    os.makedirs(output_path, exist_ok=True)

    # Initialize COCO-style dataset structure
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "card", "supercategory": "none"}]  # Modify if more categories are used
    }

    annotation_id = 0  # Unique ID for each annotation

    # Generate samples and populate the dataset
    for i in tqdm(range(n_samples), desc=f"Generating Images for {mode}"):
        bg_img_with_cards, card_infos = create_image_with_random_cards(bg_folder, cards_path)
        
        # Save the image
        img_filename = f"img_{i}.jpg"
        cv2.imwrite(output_path + f"/{img_filename}", bg_img_with_cards)

        # Add image entry
        img_id = i

        image_entry = {
            "id": img_id,
            "file_name": img_filename,
            "width": bg_img_with_cards.shape[1],  # Image width
            "height": bg_img_with_cards.shape[0],  # Image height
        }
        coco_dataset["images"].append(image_entry)

        # Add annotations for each bounding box
        for k in card_infos.keys():
            card_info = card_infos[k]
            x_min, y_min, x_max, y_max = card_info[BBOX_K] # Assuming bbox is [x_min, y_min, x_max, y_max]
            width = x_max - x_min
            height = y_max - y_min
            segmentation = card_info[COORD_K]

            segmentation = np.concatenate([s.flatten() for s in segmentation]).tolist()

            annotation_entry = {
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 0,  # Category ID for "card"
                "bbox": [x_min, y_min, width, height],  # COCO bbox format
             #   "area": width * height,  # Bounding box area
                "segmentation": segmentation,
                "iscrowd": 0  # 0 for single objects, 1 for large groups
            }
            coco_dataset["annotations"].append(annotation_entry)
            annotation_id += 1

    annotation_file = f'instances_{mode}.json'
    # Save the dataset to a JSON file
    with open(f"{annotation_path}/{annotation_file}", "w") as outfile:
        json.dump(coco_dataset, outfile, indent=4, default=convert_np_types)


if __name__ == "__main__":
    train_examples = 1000
    create_dfine_dataset(train_examples, 'train')

    val_examples = 200
    create_dfine_dataset(val_examples, 'val')

    test_examples = 800
    create_dfine_dataset(test_examples, 'test')