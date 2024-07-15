import json
import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from shutil import copy, rmtree

def download_and_process_images(json_folder: str | Path, images_folder: str | Path, labels_folder: str | Path) -> None:
    json_folder = Path(json_folder)
    images_folder = Path(images_folder)
    labels_folder = Path(labels_folder)

    # Create folders if they do not exist
    images_folder.mkdir(parents=True, exist_ok=True)
    labels_folder.mkdir(parents=True, exist_ok=True)

    # List all JSON files in the json_folder
    json_files = list(json_folder.glob('*.json'))

    # Process each JSON file
    for json_file in json_files:
        # Read the JSON file
        with open(json_file, 'r') as f:
            obj = json.load(f)

        # Filter bounding boxes to include only "Ophiuroidea"
        filtered_bounding_boxes = [
            bbox for bbox in obj.get('boundingBoxes', [])
            if bbox.get('concept') == 'Ophiuroidea'
        ]

        # If no bounding boxes are left after filtering, skip this object
        if not filtered_bounding_boxes:
            continue

        # Download the image
        response = requests.get(obj['url'])
        img = Image.open(BytesIO(response.content))

        # Save the image
        img_name = f"{obj['uuid']}.png"
        img_path = images_folder / img_name
        img.save(img_path)

        # Get image dimensions
        image_width, image_height = img.size

        # Prepare label data
        label_data = []
        for bbox in filtered_bounding_boxes:
            class_id = 0  # Assign class ID based on the concept, e.g., using a dictionary
            x_left = bbox['x']
            y_top = bbox['y']
            x_right = x_left + bbox['width']
            y_bottom = y_top + bbox['height']
            w = x_right - x_left
            h = y_bottom - y_top

            # Normalize the coordinates
            normalized_x_center = ((x_left + x_right) / 2) / image_width
            normalized_y_center = ((y_top + y_bottom) / 2) / image_height
            normalized_width = w / image_width
            normalized_height = h / image_height

            # Format the label
            text = (f'{class_id} '
                    f'{normalized_x_center} '
                    f'{normalized_y_center} '
                    f'{normalized_width} '
                    f'{normalized_height}')
            label_data.append(text)

        # Save the labels to a text file
        label_file_path = labels_folder / f"{obj['uuid']}.txt"
        with open(label_file_path, 'w') as file:
            file.write('\n'.join(label_data))

json_folder = 'json_files'
images_folder = 'images'
labels_folder = 'labels'

# Step 1: Download images and process labels
download_and_process_images(json_folder, images_folder, labels_folder)