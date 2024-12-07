import os
from tqdm import tqdm
import cv2
from make_dataset_sample import create_image_with_random_cards

if __name__ == "__main__":
    # Define folders
    bg_folder = "backgrounds"
    cards_path = "pokemon_cards"
    output_path = "yolo_dataset"
    
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
    TRAIN_SAMPLES = 1200
    VALIDATION_SAMPLES = 500
    TEST_SAMPLES = 300

    # Function to generate dataset
    def generate_samples(num_samples, prefix, image_path, label_path, is_yolo=True):
        for i in tqdm(range(num_samples), desc=f"Generating {prefix.capitalize()} Images"):
            bg_img_with_cards, bboxes = create_image_with_random_cards(
                bg_folder, cards_path, is_yolo=is_yolo
            )

            # Save image
            image_filename = f"{prefix}_img_{i}.jpg"
            cv2.imwrite(os.path.join(image_path, image_filename), bg_img_with_cards)

            # Save label file in YOLO format
            label_filename = f"{prefix}_img_{i}.txt"
            with open(os.path.join(label_path, label_filename), "w") as label_file:
                for key, bbox_data in bboxes.items():
                    bbox = bbox_data['bbox']  # YOLO format: [x_center, y_center, width, height]
                    class_id = 0  # this dataset has only one class, which is 0 (card class). Cgeck the data.yaml
                    label_file.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

    # Generate training dataset
    print("Generating training dataset...")
    generate_samples(TRAIN_SAMPLES, "train", images_train_path, labels_train_path)

    # Generate validation dataset
    print("Generating validation dataset...")
    generate_samples(VALIDATION_SAMPLES, "val", images_val_path, labels_val_path)

    # Generate test dataset
    print("Generating test dataset...")
    generate_samples(TEST_SAMPLES, "test", images_test_path, labels_test_path)

    print("Dataset creation completed.")
