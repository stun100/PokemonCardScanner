import os
from tqdm import tqdm
import cv2
from make_dataset_sample import create_image_with_random_cards
from make_dataset_util import YOLO_COORD_K, YOLO_BBOX_K
from constants import *

# Function to generate dataset
def generate_samples(num_samples, prefix, image_path, label_path, is_segmentation):
    for i in tqdm(range(num_samples), desc=f"Generating {prefix.capitalize()} Images"):
        bg_img_with_cards, cards_data = create_image_with_random_cards(
            bg_folder, cards_path
        )

        # Save image
        image_filename = f"{prefix}_img_{i}.jpg"
        cv2.imwrite(os.path.join(image_path, image_filename), bg_img_with_cards)

        class_id = 0  # this dataset has only one class, which is 0 (card class). Check data.yaml
        # Save label file in YOLO format
        label_filename = f"{prefix}_img_{i}.txt"
        with open(os.path.join(label_path, label_filename), "w") as label_file:
            if is_segmentation:
                for _, card_data in cards_data.items():
                    coords = card_data[YOLO_COORD_K]
                    label_file.write(f"{class_id} ")
                    label_file.write(" ".join([f"{coord}" for coord in coords]) + '\n')
            else:
                for _, card_data in cards_data.items():
                    bbox = card_data[YOLO_BBOX_K]  # YOLO format: [x_center, y_center, width, height]
                    range_of_bbox = range(len(bbox))
                    label_file.write(f"{class_id} ")
                    label_file.write(" ".join([str(bbox[i]) for i in range_of_bbox]) + "\n")


if __name__ == "__main__":
    # Make you own constants.py under the make_dataset directory
    bg_folder = BG_PATH
    cards_path = CARDS_PATH
    output_path = YOLO_SEG_PATH
    
    # Subfolders for train, val, and test
    images_train_path = os.path.join(output_path, "images", "train")
    images_val_path = os.path.join(output_path, "images", "val")
    images_test_path = os.path.join(output_path, "images", "test")
    labels_train_path = os.path.join(output_path, "labels", "train")
    labels_val_path = os.path.join(output_path, "labels", "val")
    labels_test_path = os.path.join(output_path, "labels", "test")

    # Create directories
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(images_test_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)
    os.makedirs(labels_test_path, exist_ok=True)

    # Dataset split configuration
    TRAIN_SAMPLES = 5000
    VALIDATION_SAMPLES = 1000
    TEST_SAMPLES = 500
    IS_SEGMENTATION = True

    # Generate training dataset
    print("Generating training dataset...")
    generate_samples(TRAIN_SAMPLES, "train", images_train_path, labels_train_path, IS_SEGMENTATION)

    # Generate validation dataset
    print("Generating validation dataset...")
    generate_samples(VALIDATION_SAMPLES, "val", images_val_path, labels_val_path, IS_SEGMENTATION)

    # Generate test dataset
    print("Generating test dataset...")
    generate_samples(TEST_SAMPLES, "test", images_test_path, labels_test_path, IS_SEGMENTATION)

    print("Dataset creation completed.")