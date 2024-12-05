import random
import os
import numpy as np
import cv2
from card_transformation import random_transform_card
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def overlay_image_alpha(background, overlay, position):
    """Overlay an image with alpha channel over a background."""
    x, y = position
    h, w = overlay.shape[:2]

    # Extract alpha channel if present
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):  # Iterate over RGB channels
            background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay[:, :, c]
    else:
        background[y:y+h, x:x+w] = overlay  # If no alpha channel, direct copy


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
        card = card_data['card']
        coords = card_data['coord']

        card_height, card_width = card.shape[:2]
        # Randomly position the card within the padded area
        x = random.randint(0, new_width - card_width)
        y = random.randint(0, new_height - card_height)

        # Translate the coordinates based on the random position
        coords = np.array(coords)  # Ensure coords is a NumPy array
        translated_coords = coords + np.array([x - padding, y - padding])

        # Update the dictionary with the translated coordinates
        cards_corner_dict[card_info]['coord'] = translated_coords.tolist()

        # Overlay the card on the padded background
        overlay_image_alpha(padded_bg, card, (x, y))

    for card_info, data in cards_corner_dict.items():
        coords = data['coord']
        updated_coords = []
        x_coords = []
        y_coords = []

        for (x, y) in coords:
            if x + padding >= new_width or y + padding >= new_height:
                updated_coords.append((-1, -1))
            else:
                updated_coords.append((x, y))
                x_coords.append(x)
                y_coords.append(y)

        # Calculate the bounding box if valid coordinates exist
        if x_coords and y_coords:
            bbox_x_min = max(0, min(x_coords))
            bbox_y_min = max(0, min(y_coords))
            bbox_x_max = min(width, max(x_coords))
            bbox_y_max = min(height, max(y_coords))
            bbox = [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]
        else:
            bbox = None  # No valid coordinates, bbox is None

        cards_corner_dict[card_info]['coord'] = updated_coords
        cards_corner_dict[card_info]['bbox'] = bbox


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
        cards_corner_dict[f'{card_id}']['card'] = transformed_card
        cards_corner_dict[f'{card_id}']['coord'] = card_corners
        transformed_cards.append(transformed_card)

    # Place the cards on the background
    bg_img_with_cards, cards_corner_dict = place_cards_on_background(bg_img, cards_corner_dict)

    # Extract coordinates and bounding boxes
    coords = [cards_corner_dict[f'{card_id}']['coord'] for card_id in cards_corner_dict.keys()]
    bboxes = [cards_corner_dict[f'{card_id}']['bbox'] for card_id in cards_corner_dict.keys()]
    other_coords = {key: {'coord': value['coord'], 'bbox': value['bbox']} for key, value in cards_corner_dict.items()}

    return [bg_img_with_cards, coords, bboxes, other_coords]

if __name__ == "__main__":
    bg_folder = "backgrounds"
    cards_path = "pokemon_cards"
    final_img, coords, bboxes, _ = create_image_with_random_cards(bg_folder, cards_path)

    # Display the final image using OpenCV's imshow
    cv2.imshow('Final Image with Cards', final_img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()  # Close the window when done


    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    # Plot the image
    plt.figure(figsize=(8, 6))
    plt.imshow(final_img_rgb)
    plt.axis('on')

    # Plot and annotate the bounding boxes
    for bbox in bboxes:
        if bbox:  # Ensure bbox is valid
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
                x_min, y_min - 10, f'({x_min}, {y_min}, {x_max}, {y_max})',
                color='yellow',
                fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5)
            )

    # Display the result
    plt.title("Final Image with Bounding Boxes")
    plt.show()