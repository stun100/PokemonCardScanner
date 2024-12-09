import numpy as np
import cv2

def reorder_points(points):
    # Calculate centroids
    centroids = np.mean(points, axis=0)

    # Sort points based on distance from centroids
    points_sorted = sorted(points, key=lambda x: np.arctan2(x[0][1] - centroids[0][1], x[0][0] - centroids[0][0]))

    return np.array(points_sorted)

def get_card_dimensions(corners):
    # Calculate the width and height of the card
    width = np.linalg.norm(corners[0] - corners[1])
    height = np.linalg.norm(corners[1] - corners[2])
    return width, height

def perspective_transform(image, mask, mirror=False):
    # Convert the boolean mask to uint8
    mask_uint8 = mask.astype(np.uint8) * 255

    # Visualize the mask
    h, w = mask_uint8.shape
    mask_visualized = np.zeros((h, w, 3), dtype=np.uint8)
    # Set the background pixels to black
    mask_visualized[:, :] = [0, 0, 0]
    # Set the object mask pixels to red
    mask_visualized[mask_uint8 != 0] = [0, 0, 255]
    # viewer.VideoFrameBuilder().add_image(mask_visualized, 2, "Segmentation Mask")

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_detections = image.copy()

    # Approximate the contours to get the four corners of the card
    for contour in contours:
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            # Rearrange the points in the approx array if needed
            approx = reorder_points(approx)

            # Draw the contour on the image
            cv2.drawContours(image_with_detections, [approx], -1, (0, 0, 255), 2)

            # Draw the corners on the image
            circle_size = 8
            for point in approx:
                x, y = point.ravel()
                cv2.circle(image_with_detections, (x, y), circle_size, (0, 255, 0), -1)

            # viewer.VideoFrameBuilder().add_image(image_with_detections, 2, "Edge & Corner Detection")

            # Perform perspective transform
            width, height = get_card_dimensions(approx)
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(approx.astype(np.float32), dst)

            warped = cv2.warpPerspective(image, M, (int(width), int(height)))

            # Rotate the warped image to ensure portrait orientation
            if width > height:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            # Resize the warped image to fill the canvas without maintaining aspect ratio
            warped_stretched = cv2.resize(warped, (320, 320))

            if mirror is True:
                warped_stretched = cv2.flip(warped_stretched, 1)

            return warped_stretched

    return None
