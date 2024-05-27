from typing import Tuple

import cv2
import numpy as np
from cv2.typing import MatLike
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def augment_images(images: np.array, labels: np.array, augmentations_per_image: int = 5) -> Tuple[np.array, np.array]:
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    augmented_labels = []

    for i in range(len(images)):
        img = images[i]
        label = labels[i]
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        aug_count = 0
        for batch in datagen.flow(img, batch_size=1):
            aug_img = batch[0].astype('float32').squeeze()
            augmented_images.append(aug_img)
            augmented_labels.append(label)
            aug_count += 1
            if aug_count >= augmentations_per_image:
                break

    return np.array(augmented_images), np.array(augmented_labels)


def augment_image_example(image: MatLike, augmentations_per_image: int = 5) -> np.array:
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.5,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    original_shape = image.shape[:2]
    image = cv2.resize(image, (original_shape[1], original_shape[0]))

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    augmented_images = []

    for i, batch in enumerate(datagen.flow(image, batch_size=1)):
        aug_img = batch[0].astype('uint8').squeeze()
        aug_img = cv2.resize(aug_img, (original_shape[1], original_shape[0]))
        augmented_images.append(aug_img)
        if i >= augmentations_per_image - 1:
            break

    return augmented_images
