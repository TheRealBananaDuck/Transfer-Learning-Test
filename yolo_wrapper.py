import warnings
from shutil import copy, rmtree
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import matplotlib.pyplot as plt

class YoloWrapper:
    def __init__(self, model_weights: str) -> None:
        """
        Initialize YOLOv8 model with weights.
        Args:
            model_weights (str): model weight can be one of the follows:
                - 'nano' for YOLOv8 nano model
                - 'small' for YOLOv8 small model
                - a path to a .pt file contains the weights from a previous training.
        """
        if model_weights == 'nano':
            model_weights = 'yolov8n.pt'
        elif model_weights == 'small':
            model_weights = 'yolov8s.pt'
        elif model_weights == 'medium':
            model_weights = 'yolov8m.pt'
        elif (not Path(model_weights).exists()) or (Path(model_weights).suffix != '.pt'):
            raise ValueError('The parameter model_weight should be "nano", "small" or a'
                             'path to a .pt file with saved weights')

        # initialize YOLO model
        self.model = YOLO(model_weights)

    def create_dataset(images_path: str | Path, labels_path: str | Path = None, result_path: str | Path = None,
        train_size: float = 0.9) -> None:
        """
        Create A YOLO dataset from a folder of images and a folder of labels. The function
        assumes all the images have a labels with the same name. The output structure is
        - result_path
            - images
                - train
                - val (optional)
            - labels
                - train
                - val (optional)
        Args:
            images_path (str|Path): path to the folder contains the images
            labels_path (str|Path): path to the folder contains the labels
            result_path (optional, str|Path): path to the folder where the result will be saved.
                If it's None, a folder named 'data' will be created in parent directory of the images.
            train_size (float): a number between 0 and 1 represent the proportion of the dataset to
                include in the train split

        Returns:

        """
        if train_size <= 0 or 1 < train_size:
            raise ValueError(f'Train size should be between 0 to 1, but got {train_size}')

        images_path = Path(images_path)
        labels_path = Path(labels_path)

        if result_path is None:
            parent_dir = images_path.parent
            result_path = parent_dir / 'data'
        else:
            result_path = Path(result_path)

        if result_path.exists():
            rmtree(result_path)

        all_images = sorted(list(images_path.glob('*')))
        all_labels = sorted(list(labels_path.glob('*')))

        training_dataset, val_dataset, train_labels, val_labels = train_test_split(
            all_images, all_labels, train_size=train_size)

        result_path_image_training = result_path / 'images' / 'train'
        result_path_image_training.mkdir(parents=True, exist_ok=False)
        result_path_label_training = result_path / 'labels' / 'train'
        result_path_label_training.mkdir(parents=True, exist_ok=False)

        for image, label in zip(training_dataset, train_labels):
            copy(image, result_path_image_training / image.name)
            copy(label, result_path_label_training / label.name)

        if val_dataset:
            result_path_image_validation = result_path / 'images' / 'val'
            result_path_image_validation.mkdir(parents=True, exist_ok=False)
            result_path_label_validation = result_path / 'labels' / 'val'
            result_path_label_validation.mkdir(parents=True, exist_ok=False)

            for image, label in zip(val_dataset, val_labels):
                copy(image, result_path_image_validation / image.name)
                copy(label, result_path_label_validation / label.name)

    def create_config_file(parent_data_path: str | Path, class_names: list[str], path_to_save: str = None) -> None:
            
            """
            Create YOLOv8 configuration yaml file. The configuration file contains:
            path -  absolute path to the folder contains the images and labels folders with the data
            train - relative path to 'path' of the train images folder (images/train)
            val -  relative path to 'path' of the validation images folder (images/val), if exists
            nc - the number of classes
            names - a list of the classes names
            Args:
                parent_data_path (str|Path): path to the folder contains the images and labels folder with the data.
                    The structure of this folder should be:
                    - parent_data_path
                        - images
                            - train
                            - val (optional)
                        - labels
                            - train
                            - val (optional)
                class_names (list[str]): a list contains the names of the classes. The first name is for label 0, and so on
                path_to_save (Optional, str): A path to where to save the result. By defulat it save it in the working
                    directory as 'config.yaml'. If a folder is given a file 'config.yaml' will be saved inside. If a path
                    including file name is given, the file must be with a .yaml suffix.

            Returns:

            """
            parent_data_path = Path(parent_data_path)
            if not parent_data_path.exists():
                raise FileNotFoundError(f'Folder {parent_data_path} is not found')
            if not (parent_data_path / 'images' / 'train').exists():
                raise FileNotFoundError(f'There is not folder {parent_data_path / "images" / "train"}')
            if not (parent_data_path / 'labels' / 'train').exists():
                raise FileNotFoundError(f'There is not folder {parent_data_path / "labels" / "train"}')

            config = {
                'path': str(parent_data_path.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': len(class_names),
                'names': class_names
            }

            if not (parent_data_path / 'images' / 'val').exists():
                config.pop('val')

            if path_to_save is None:
                path_to_save = 'config.yaml'
            path_to_save = Path(path_to_save)

            if not path_to_save.suffix:  # is a folder
                path_to_save.mkdir(parents=True, exist_ok=True)
                path_to_save = path_to_save / 'config.yaml'

            if path_to_save.suffix != '.yaml':
                raise ValueError(f'The path to save the configuration file should be a folder, a yaml file or None.'
                                f'Got a {path_to_save.suffix} file instead')

            with open(path_to_save, 'w', encoding = 'utf8') as file:
                for key, value in config.items():
                    file.write(f'{key}: {value}\n')

    def train(self, config: str, epochs: int = 100, name: str = None) -> None:
        """
        Train the model. After running a 'runs/detect/<name>' folder will be created and stores information
        about the training and the saved weights.
        Args:
            config (str): a path to a configuration yaml file for training.
                Such a file contains:
                    path -  absolute path to the folder contains the images and labels folders with the data
                    train - relative path to 'path' of the train images folder (images/train)
                    val -  relative path to 'path' of the validation images folder (images/val), if exists
                    nc - the number of classes
                    names - a list of the classes names
                Can be created with the create_config_file method.
            epochs (int): number of epochs for training
            name (str): the name of the results' folder. If None (default) a default name 'train #' will
                be created.

        Returns:

        """
        if Path(config).suffix != '.yaml':
            raise ValueError('Config file should be a yaml file')
        self.model.train(data=config, epochs=epochs, name=name, freeze=10)

    def predict_and_show(self, image: str | np.ndarray, threshold: float = 0.25) -> None:
        """
        Predict bounding box for a single image and show the bounding box with its confidence.
        Args:
            image (str | np.ndarray): a path to an image or a BGR np.ndarray image to predict
                bounding box for
            threshold (float): a number between 0 and 1 for the confidence a bounding box should have to
                consider as a detection. Default is 0.25.
        Returns:

        """
        yolo_results = self.model(image, threshold=threshold)
        labeled_image = yolo_results[0].plot()
        plt.figure()
        plt.imshow(labeled_image[..., ::-1])  # change channels order since the YOLO work on BGR images
        plt.show()
