import json
from tqdm import tqdm 
import cv2
from make_dataset_sample import create_image_with_random_cards 
from constants import CARDS_PATH, BG_PATH, OUTPUT_PATH

if __name__ == "__main__":
    # Make you own constants.py under the make_dataset directory
    bg_folder = BG_PATH
    cards_path = CARDS_PATH
    output_path = OUTPUT_PATH

    dataset = {}

    NUM_SAMPLES = 1000

    # Add tqdm to the loop
    for i in tqdm(range(NUM_SAMPLES), desc="Generating Images"):
        bg_img_with_cards, bboxes = create_image_with_random_cards(bg_folder, cards_path)
        
        dataset[f"img_{i}.jpg"] = bboxes
        cv2.imwrite(output_path + f"/img_{i}.jpg", bg_img_with_cards)

    with open(f"{output_path}/dataset.json", "w") as outfile: 
        json.dump(dataset, outfile)