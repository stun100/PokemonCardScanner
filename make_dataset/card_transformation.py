import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from constants import CARDS_PATH

def randomize_contrast_brightness(img, contrast_range=(0.8, 1.2), brightness_range=(-75, 75)):
    # Randomize contrast and brightness values
    if img.shape[2] == 4:  # RGBA
        rgb = img[:, :, :3]
        alpha = img[:, :, 3]
    else:  # RGB
        rgb = img
        alpha = None

    # Randomize contrast and brightness
    contrast = random.uniform(*contrast_range)
    brightness = random.randint(*brightness_range)

    # Apply adjustments only to the RGB channels
    rgb = np.int16(rgb)
    rgb = rgb * contrast + brightness
    rgb = np.clip(rgb, 0, 255)
    rgb = np.uint8(rgb)

    # Combine RGB and alpha channels if alpha exists
    if alpha is not None:
        adjusted_img = np.dstack((rgb, alpha))
    else:
        adjusted_img = rgb

    return adjusted_img

def random_scaling(img, bg_width, bg_height):
    scale = random.uniform(0.3, 0.45) # Random scaling factor between 0.5 and 1.5
    scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    while (scaled_img.shape[0] >= bg_width // 2 and scaled_img.shape[1] >= bg_height // 2): # we want cards that take at most half the background
        scale = random.uniform(0.45, 0.65) # Random scaling factor between 0.5 and 1.5
        scaled_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    return scaled_img

import numpy as np
import cv2
import random

def random_skew(img):
    height, width, _ = img.shape
    # Define random skew
    margin = int(0.2 * min(width, height))
    src_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    dst_points = np.float32([
        [random.randint(0, margin), random.randint(0, margin)],  # Top-left
        [width - random.randint(0, margin), random.randint(0, margin)],  # Top-right
        [width - random.randint(0, margin), height - random.randint(0, margin)],  # Bottom-right
        [random.randint(0, margin), height - random.randint(0, margin)]  # Bottom-left
    ])

    # Perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    skewed_img = cv2.warpPerspective(img, matrix, (width, height))

    # Apply the transformation to the corners to get the new positions
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    new_corners = cv2.perspectiveTransform(np.array([corners]), matrix)[0]

    # Sort corners clockwise
    # Step 1: Sort by y-coordinate (ascending)
    sorted_by_y = sorted(new_corners, key=lambda x: x[1])
    # Step 2: Identify top-left and top-right
    top_two = sorted_by_y[:2]
    top_left = min(top_two, key=lambda x: x[0])
    top_right = max(top_two, key=lambda x: x[0])
    # Step 3: Identify bottom-left and bottom-right
    bottom_two = sorted_by_y[2:]
    bottom_left = min(bottom_two, key=lambda x: x[0])
    bottom_right = max(bottom_two, key=lambda x: x[0])

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])

    # Return both skewed image and new corners in clockwise order
    return skewed_img, ordered_corners


def random_rotate(img, corners, angle_range=(0, 360)):
    # rotates the card
    angle = random.uniform(*angle_range)
    center = (img.shape[1] // 2, img.shape[0] // 2)

    # Calculate the bounding box size for the rotated image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # Compute new bounding box dimensions
    new_width = int(img.shape[0] * sin + img.shape[1] * cos)
    new_height = int(img.shape[0] * cos + img.shape[1] * sin)

    # Apply the rotation with the new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    ones = np.ones((corners.shape[0], 1))
    homogeneous_corners = np.hstack([corners, ones])
    transformed_corners = np.dot(rotation_matrix, homogeneous_corners.T).T

    return rotated_img, transformed_corners.astype(int)

def random_transform_card(cards_path, cards_picked, bg_width=480, bg_height=480):
    # Load random card from the directory
    card_file = random.choice(list(set(os.listdir(cards_path)).difference(cards_picked)))
    card_id = card_file.split(".")[0]
    cards_picked.add(card_file)
    card_path = os.path.join(cards_path, card_file)

    image = cv2.imread(card_path, cv2.IMREAD_UNCHANGED)
    if image.shape[2] != 4:
        b, g, r = cv2.split(image)
        alpha = np.ones_like(b, dtype=np.uint8) * 255  # Fully opaque alpha channel
        image = cv2.merge((b, g, r, alpha))

    scaled_img = random_scaling(image, bg_width, bg_height)

    skewed_img, corners = random_skew(scaled_img)

    # Perform random rotation and resize to ensure no cutoff
    rotated_img, corners = random_rotate(skewed_img, corners)

    jittered_img = randomize_contrast_brightness(rotated_img)

    # Ensure the output is in RGBA format
    # output = cv2.cvtColor(jittered_img, cv2.COLOR_BGRA2RGBA)

    return [jittered_img, corners, card_id]

if __name__ == "__main__":
    # Make you own constants.py under the make_dataset directory
    cards_path = CARDS_PATH

    transformed_card, corners, _ = random_transform_card(cards_path, set())

    cv2.imshow('Result', transformed_card)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    transformed_card = cv2.cvtColor(transformed_card, cv2.COLOR_BGRA2RGBA)
    plt.figure(figsize=(8, 6))
    plt.imshow(transformed_card)
    plt.axis('on')

    # Plot and annotate the corners
    for idx, (x, y) in enumerate(corners):
        plt.scatter(x, y, color='red', s=5)  # Mark the corner point
        plt.text(x + 5, y - 10, f'({x}, {y})', color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5))

    # Display the result
    plt.title("Transformed Card with Corner Annotations")
    plt.show()