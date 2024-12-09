import os
import cv2

def center_crop_background(bg_path, output_dir, min_width=640, min_height=640):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    processed_count = 0

    # Loop through all images in the bg_path
    for filename in os.listdir(bg_path):
        file_path = os.path.join(bg_path, filename)

        # Check if it's a file and has a valid image extension
        if not os.path.isfile(file_path) or not filename.lower().endswith(('jpg')):
            continue

        # Read the image
        image = cv2.imread(file_path)

        if image is None:
            print(f"Failed to read image {filename}. Skipping...")
            continue

        # Get image dimensions
        height, width, _ = image.shape

        # Skip if dimensions are smaller than the minimum
        if width < min_width or height < min_height:
            continue

        # Calculate center crop coordinates
        start_x = (width - min_width) // 2
        start_y = (height - min_height) // 2
        end_x = start_x + min_width
        end_y = start_y + min_height

        # Perform the crop
        cropped_image = image[start_y:end_y, start_x:end_x]

        # Save the cropped image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped_image)

        # Increment the counter
        processed_count += 1

        # Print a message every 1000 images
        if processed_count % 1000 == 0:
            print(f"Saved {processed_count} images.")

        
#Example usage
bg_path = 'C:/aml_dataset/visualgenome'
output_dir = 'C:/aml_dataset/backgrounds640'
center_crop_background(bg_path, output_dir)