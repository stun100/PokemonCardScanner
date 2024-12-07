import numpy as np

CARD_K = 'card'
YOLO_BBOX_K = 'yolo_bbox_k'
BBOX_K = 'bbox'
COORD_K = 'coord'
YOLO_COORD_K = 'yolo_coord_k'

### Translates the coordinates of the corner of the cards, that can be randomly placed outside, inside the background
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
        visible_points = np.array(visible_points, dtype=int)
        center = visible_points.mean(axis=0)
        visible_points = sorted(visible_points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))

    normalized_points = [(x / width, y / height) for x, y in visible_points]
    visible_points_yolo = [coord for point in normalized_points for coord in point]
    
    return visible_points, visible_points_yolo

# Define the bounding boxes (with and without yolo format) of a card given its coordinates. 
def define_bounding_box(coords, height, width):
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    bbox_x_min = max(0, min(x_coords))
    bbox_y_min = max(0, min(y_coords))
    bbox_x_max = min(width, max(x_coords))
    bbox_y_max = min(height, max(y_coords))

    x_center = (bbox_x_min + bbox_x_max) / 2.0 / width
    y_center = (bbox_y_min + bbox_y_max) / 2.0 / height
    bbox_width = (bbox_x_max - bbox_x_min) / width
    bbox_height = (bbox_y_max - bbox_y_min) / height

    bbox = [bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max]
    yolo_bbox = [x_center, y_center, bbox_width, bbox_height]
    
    return bbox, yolo_bbox

# Overlays the card to the background
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