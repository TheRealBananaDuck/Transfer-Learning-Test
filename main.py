import json
import os
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from shutil import copy, rmtree
import numpy as np
import matplotlib as plt
from pathlib import Path
import cv2
from yolo_wrapper import YoloWrapper



# paths to the data
dataset_path = Path('yolo_dataset')  # where the YOLO dataset will be
large_field_images_path = Path('image')  # where the original images
labels_path = Path('labels')  # where the labels are


# create the dataset in the format of YOLO
YoloWrapper.create_dataset(large_field_images_path, labels_path, dataset_path)
# create YOLO configuration file
config_path = 'brittle_star_config.yaml'
YoloWrapper.create_config_file(dataset_path, ['brittle_star'], config_path)

# create pretrained YOLO model and train it using transfer learning
model = YoloWrapper('nano')
model.train(config_path, epochs=200, name='blood_cell')

# make predictions on the validation set
data_to_predict_path = dataset_path/'images'/'val'
val_image_list = list(data_to_predict_path.glob('*.jpg'))

# save the prediction in a csv file where the bounding boxes should have minimum size
model.predict_and_show()



