import json
from tqdm import tqdm 
import cv2
from make_dataset_sample import create_image_with_random_cards 

if __name__ == "__main__":
    bg_folder = "backgrounds"
    cards_path = "pokemon_cards"
    output_path = "output"

    dataset = {}

    NUM_SAMPLES = 1000

    # Add tqdm to the loop
    for i in tqdm(range(NUM_SAMPLES), desc="Generating Images"):
        bg_img_with_cards, coords, bboxes, other_coords = create_image_with_random_cards(bg_folder, cards_path)
        
        dataset[f"img_{i}.jpg"] = bboxes
        cv2.imwrite(output_path + f"/img_{i}.jpg", bg_img_with_cards)

    with open(f"{output_path}/dataset.json", "w") as outfile: 
        json.dump(dataset, outfile)