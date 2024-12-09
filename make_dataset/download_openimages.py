import pandas as pd
import requests
import os
from PIL import Image
import io
from tqdm import tqdm  # Import tqdm for progress bar

# Path to metadata CSV
metadata_path = 'make_dataset/test-images-with-rotation.csv'  # Replace with actual path

# Load metadata
metadata = pd.read_csv(metadata_path)
print(metadata.columns)

# Output directory for filtered images
output_dir = 'datasets/new_backgrounds'
os.makedirs(output_dir, exist_ok=True)

# Counter for downloaded images
downloaded_count = 0
max_files = 8000

# Iterate through metadata with a progress bar
for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Downloading Images"):
    if downloaded_count >= max_files:
        print("Reached the limit of 15,000 files. Stopping download.")
        break

    image_url = row['OriginalURL']
    image_id = row['ImageID']
    try:
        response = requests.get(image_url, stream=True, timeout=10)  # Added timeout for stability
        if response.status_code == 200:
            # Open image and check size
            img = Image.open(io.BytesIO(response.content))
            if img.width >= 640 and img.height >= 640:
                # Save the image
                with open(os.path.join(output_dir, f"{image_id}.jpg"), 'wb') as f:
                    f.write(response.content)
                downloaded_count += 1  # Increment counter
    except Exception as e:
        print(f"Failed to process {image_id}: {e}")

print(f"Downloaded {downloaded_count} images.")
