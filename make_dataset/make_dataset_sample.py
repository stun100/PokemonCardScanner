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

def place_cards_on_background_yolo(bg_img, cards_corner_dict):
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

            # Convert to YOLO format (normalized values)
            x_center = (bbox_x_min + bbox_x_max) / 2.0 / width
            y_center = (bbox_y_min + bbox_y_max) / 2.0 / height
            bbox_width = (bbox_x_max - bbox_x_min) / width
            bbox_height = (bbox_y_max - bbox_y_min) / height

            bbox = [x_center, y_center, bbox_width, bbox_height]
        else:
            bbox = None  # No valid coordinates, bbox is None

        cards_corner_dict[card_info]['coord'] = updated_coords
        cards_corner_dict[card_info]['bbox'] = bbox

    # Extract the central part of the padded image, corresponding to the original size
    output = padded_bg[padding:padding + height, padding:padding + width]

    return output, cards_corner_dict

def place_cards_on_background_yolo_segmentation(bg_img, cards_corner_dict):
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

    def calculate_intersection_points(bg_img, coords):
        """Clip the polygon to fit inside the image and return visible points."""
        height, width = bg_img.shape[:2]

        # Define boundaries as lines
        boundaries = {
            "top": ((0, 0), (width - 1, 0)),        # Top: y = 0
            "bottom": ((0, height - 1), (width - 1, height - 1)),  # Bottom: y = height-1
            "left": ((0, 0), (0, height - 1)),     # Left: x = 0
            "right": ((width - 1, 0), (width - 1, height - 1))     # Right: x = width-1
        }

        def line_intersection(line1, line2):
            """Calculate intersection point of two lines."""
            x1, y1, x2, y2 = *line1[0], *line1[1]
            x3, y3, x4, y4 = *line2[0], *line2[1]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None  # Lines are parallel or coincident
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return px, py

        def is_point_on_segment(point, segment):
            """Check if a point is on a segment."""
            px, py = point
            (x1, y1), (x2, y2) = segment
            return (min(x1, x2) <= px <= max(x1, x2)) and (min(y1, y2) <= py <= max(y1, y2))

        # Generate edges from the card's coordinates
        edges = [(coords[i], coords[(i + 1) % len(coords)]) for i in range(len(coords))]

        # Find intersections with boundaries
        visible_points = []
        for edge in edges:
            for boundary in boundaries.values():
                intersection = line_intersection(edge, boundary)
                if intersection and is_point_on_segment(intersection, edge) and is_point_on_segment(intersection, boundary):
                    visible_points.append(intersection)

        # Include original points that are inside the image
        visible_points.extend([(x, y) for x, y in coords if 0 <= x < width and 0 <= y < height])

        # Sort the visible points to form a polygon
        if visible_points:
            visible_points = np.array(visible_points)
            center = visible_points.mean(axis=0)
            visible_points = sorted(visible_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))

        return visible_points

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

        # Clip the card's polygon to the image boundaries
        intersection_points = calculate_intersection_points(bg_img, translated_coords)

        # Update the dictionary with visible points
        cards_corner_dict[card_info]['visible_points'] = intersection_points

        # Overlay the card on the padded background
        overlay_image_alpha(padded_bg, card, (x, y))

    # Extract the central part of the padded image, corresponding to the original size
    output = padded_bg[padding:padding + height, padding:padding + width]

    return output, cards_corner_dict


def create_image_with_random_cards(bg_folder, cards_path, num_cards_range=(1, 1), cards_picked=set(), is_yolo=False):
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
    if is_yolo:
        bg_img_with_cards, cards_corner_dict = place_cards_on_background_yolo(bg_img, cards_corner_dict)
    else:
        bg_img_with_cards, cards_corner_dict = place_cards_on_background(bg_img, cards_corner_dict)

    # Extract coordinates, bounding boxes, and corners
    card_data = {
        key: {
            'bbox': value['bbox'],
            'coord': value['coord']
        } for key, value in cards_corner_dict.items()
    }

    return bg_img_with_cards, card_data


def create_image_with_random_cards2(bg_folder, cards_path, num_cards_range=(1, 5), cards_picked=set()):
    import os
    import random
    import cv2

    # Load all background images from bg_folder
    bg_files = os.listdir(bg_folder)
    bg_file = random.choice(bg_files)
    bg_path = os.path.join(bg_folder, bg_file)

    # Open background image (ensure it's in RGBA format for transparency handling)
    bg_img = cv2.imread(bg_path)

    # Determine how many cards to place on the background (randomly between num_cards_range)
    num_cards = random.randint(*num_cards_range)

    # Create a list of transformed cards and their metadata
    transformed_cards = []
    cards_corner_dict = {}
    for _ in range(num_cards):
        transformed_card, card_corners, card_id = random_transform_card(cards_path, cards_picked)
        cards_corner_dict[f'{card_id}'] = {
            'card': transformed_card,
            'coord': card_corners
        }
        transformed_cards.append(transformed_card)

    # Place the cards on the background
    bg_img_with_cards, cards_corner_dict = place_cards_on_background_yolo_segmentation(bg_img, cards_corner_dict)

    # Extract visible points for debugging or further processing
    visible_points_data = {
        key: {'visible_points': value.get('visible_points', [])} for key, value in cards_corner_dict.items()
    }

    return [bg_img_with_cards, visible_points_data]


if __name__ == "__main__":
    bg_folder = "backgrounds"
    cards_path = "pokemon_cards"

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

    print(cards_data)

    # Plot and annotate the bounding boxes and corners
    for card_id in cards_data:
        data = cards_data[card_id]
        bbox = data.get("bbox")
        coords = data.get("coord")

        # Plot bounding box if available
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
