import os
from typing import Tuple

import cv2
import numpy as np


def load_images_from_folder(folder_path: str, image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.array, np.array]:
    images = []
    labels = []

    for folder_name in os.listdir(folder_path):
        folder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, dsize=image_size)
                images.append(image)
                labels.append(folder_name)

    return np.array(images), np.array(labels)


def normalize_images(images: np.array) -> np.array:
    normalized_images = [img / 255.0 for img in images]
    return normalized_images


def generate_histogram(image: np.array) -> np.array:
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()
    return histogram
