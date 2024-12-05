import json
from tqdm import tqdm  # Import tqdm
import cv2  # Ensure cv2 is imported for image saving
from make_dataset_sample import create_image_with_random_cards  # Import numpy for grid calculations

bg_folder = "backgrounds"
cards_path = "pokemon_cards"
output_path = "output"

dataset = {}

NUM_SAMPLES = 1000

# Add tqdm to the loop
for i in tqdm(range(NUM_SAMPLES), desc="Generating Images"):
    final_img, _, card_dict = create_image_with_random_cards(bg_folder, cards_path)
    
    dataset[f"img_{i}.jpg"] = card_dict
    cv2.imwrite(output_path + f"/img_{i}.jpg", final_img)

with open(f"{output_path}/dataset.json", "w") as outfile: 
    json.dump(dataset, outfile)