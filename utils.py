import cv2
import os
from tqdm import tqdm

def resize_images_to_smallest(cards_path, output_path):
    """
    Resizes all images in the specified directory to the smallest dimensions found among them,
    ensuring transparency (alpha channel) is preserved if present.

    Parameters:
        cards_path (str): Path to the directory containing the images.
    """
    # List to store dimensions of all images
    # dimensions = []
    image_files = [f for f in os.listdir(cards_path) if os.path.isfile(os.path.join(cards_path, f))]

    # Read all images and get their dimensions
    # for file_name in tqdm(image_files, desc="Finding smallest dimensions"):
    #     file_path = os.path.join(cards_path, file_name)
    #     img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Read with alpha channel if present
    #     if img is not None:
    #         dimensions.append((img.shape[1], img.shape[0]))  # (width, height)

    # Find the smallest dimensions
    min_width = 400
    min_height = 550

    # Resize all images to the smallest dimensions
    for file_name in tqdm(image_files, desc="Resizing images"):
        print("x")
        file_path = os.path.join(cards_path, file_name)
        file_out_path = os.path.join(output_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if len(img.shape) == 3:  # If the image has no alpha channel
                resized_img = cv2.resize(img, (min_width, min_height), interpolation=cv2.INTER_AREA)
            elif len(img.shape) == 4:  # If the image has an alpha channel
                b, g, r, a = cv2.split(img)  # Split into color and alpha channels
                resized_b = cv2.resize(b, (min_width, min_height), interpolation=cv2.INTER_AREA)
                resized_g = cv2.resize(g, (min_width, min_height), interpolation=cv2.INTER_AREA)
                resized_r = cv2.resize(r, (min_width, min_height), interpolation=cv2.INTER_AREA)
                resized_a = cv2.resize(a, (min_width, min_height), interpolation=cv2.INTER_AREA)
                resized_img = cv2.merge((resized_b, resized_g, resized_r, resized_a))  # Merge back

            cv2.imwrite(file_out_path, resized_img)  

    print(f"All images resized to {min_width}x{min_height}.")

# Example usage:
resize_images_to_smallest("pokemon_cards", "pokemon_cards_resize")
