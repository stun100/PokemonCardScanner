import random
import os
import numpy as np
import cv2

from card_transformation import random_transform_card

# importing utils functions
from make_dataset_util import calculate_intersection_points, define_bounding_box, overlay_image_alpha
# importing utils constants
from make_dataset_util import CARD_K, BBOX_K, YOLO_BBOX_K, COORD_K

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def place_cards_on_background(bg_img, cards_corner_dict):
    height, width = bg_img.shape[:2]

    # Ensure the background image is in RGBA format
    if bg_img.shape[2] != 4:  # Check if the image has 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    padding = height // 6  # Adjust padding as per your requirement

    new_width = width + 2 * padding
    new_height = height + 2 * padding

    # Create a new padded background
    padded_bg = np.zeros((new_height, new_width, 4), dtype=bg_img.dtype)
    
    # Place the original image in the center of the padded background
    padded_bg[padding:padding + height, padding:padding + width] = bg_img

    for i, (card_info, card_data) in enumerate(cards_corner_dict.items()):
        card = card_data[CARD_K]
        coords = card_data[COORD_K]

        card_height, card_width = card.shape[:2]
        # Randomly position the card within the padded area
        x = random.randint(0, new_width - card_width)
        y = random.randint(0, new_height - card_height)

        # Translate the coordinates based on the random position
        coords = np.array(coords)  # Ensure coords is a NumPy array
        translated_coords = coords + np.array([x - padding, y - padding])

        # Update the dictionary with the translated coordinates
        cards_corner_dict[card_info][COORD_K] = translated_coords.tolist()
   
        intersection_points = calculate_intersection_points(bg_img, translated_coords)
        cards_corner_dict[card_info][COORD_K] = intersection_points

        # define the bounding box (both with and without YOLO format)
        bbox, yolo_bbox = define_bounding_box(intersection_points, height, width)
        cards_corner_dict[card_info][BBOX_K] = bbox
        cards_corner_dict[card_info][YOLO_BBOX_K] = yolo_bbox

        # Overlay the card on the padded background
        overlay_image_alpha(padded_bg, card, (x, y))

    # Extract the central part of the padded image, corresponding to the original size
    output = padded_bg[padding:padding + height, padding:padding + width]

    return output, cards_corner_dict


def create_image_with_random_cards(bg_folder, cards_path, num_cards_range=(1, 5), cards_picked=set()):
    # Load all background images from bg_folder
    bg_files = os.listdir(bg_folder)
    bg_file = random.choice(bg_files)
    bg_path = os.path.join(bg_folder, bg_file)

    # Open background image (make sure it's in RGBA format for transparency handling)
    bg_img = cv2.imread(bg_path)

    # Determine how many cards to place on the background (randomly between 1 and 5)
    num_cards = random.randint(*num_cards_range)

    # Create a list of transformed cards
    transformed_cards = []
    cards_corner_dict = {}
    for _ in range(num_cards):
        transformed_card, card_corners, card_id = random_transform_card(cards_path, cards_picked)
        cards_corner_dict[f'{card_id}'] = {}
        cards_corner_dict[f'{card_id}'][CARD_K] = transformed_card
        cards_corner_dict[f'{card_id}'][COORD_K] = card_corners
        transformed_cards.append(transformed_card)

    # Place the cards on the background
    bg_img_with_cards, cards_corner_dict = place_cards_on_background(bg_img, cards_corner_dict)

    # Extract coordinates, bounding boxes, and corners
    card_data = {
        key: {
            BBOX_K: value[BBOX_K],
            YOLO_BBOX_K: value[YOLO_BBOX_K],
            COORD_K: value[COORD_K]
        } for key, value in cards_corner_dict.items()
    }

    return bg_img_with_cards, card_data

if __name__ == "__main__":
    bg_folder = "C:\\aml_dataset\\backgrounds"
    cards_path = "C:\\aml_dataset\\pokemon_cards"

    final_img, cards_data = create_image_with_random_cards(bg_folder, cards_path)

    # Display the final image using OpenCV's imshow
    cv2.imshow('Final Image with Cards', final_img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window when done

    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.figure(figsize=(10, 8))
    plt.imshow(final_img_rgb)
    plt.axis('on')

    # Plot and annotate the bounding boxes and corners
    for card_id in cards_data:
        data = cards_data[card_id]
        bbox = data.get(BBOX_K)
        yolo_bbox = data.get(YOLO_BBOX_K)
        coords = data.get(COORD_K)

        print(f'The YOLO bounding box for the card {card_id} is {yolo_bbox}')

        x_min, y_min, x_max, y_max = bbox
        # Create a rectangle patch for the bounding box
        rect = patches.Rectangle(
            (x_min, y_min),  # Bottom-left corner
            x_max - x_min,  # Width
            y_max - y_min,  # Height
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        plt.gca().add_patch(rect)
        # Annotate the bounding box
        plt.text(
            x_min, y_min - 10, f'BBox: ({x_min}, {y_min}, {x_max}, {y_max})',
            color='yellow',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.5)
        )

        # Plot corners
        if coords:  # Ensure coordinates are valid
            for (x, y) in coords:
                if (x, y) != (-1, -1):  # Ignore invalid corners
                    plt.scatter(x, y, color='cyan', s=50)  # Plot corners as points
                    plt.text(
                        x + 5, y + 5, f'({x}, {y})',  # Slight offset for readability
                        color='cyan',
                        fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5)
                    )

    # Display the result
    plt.title("Final Image with Bounding Boxes and Corners")
    plt.show()
