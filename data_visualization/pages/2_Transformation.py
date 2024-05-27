import sys

import cv2
import streamlit as st
import os
import pandas as pd

from src.preprocessing import generate_histogram

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.augment import augment_image_example

# Configuration de la page
st.set_page_config(layout="wide")

st.title("Visualisation et Prétraitement des Images de Radiographies")

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))


@st.cache_data
def get_folders(path):
    if os.path.exists(path):
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    return []


@st.cache_data
def get_image_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]


@st.cache_resource
def load_image(image_path):
    return cv2.imread(image_path)


@st.cache_data
def process_image(image, rotate=0, zoom=1.0, brightness=1.0, contrast=1.0, new_width=None, new_height=None,
                  flip_horizontal=False, flip_vertical=False):
    transformed_image = image.copy()

    if rotate != 0:
        center = (transformed_image.shape[1] // 2, transformed_image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
        transformed_image = cv2.warpAffine(transformed_image, matrix,
                                           (transformed_image.shape[1], transformed_image.shape[0]))

    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)

    if flip_vertical:
        transformed_image = cv2.flip(transformed_image, 0)

    if zoom != 1.0:
        height, width = transformed_image.shape[:2]
        new_width_zoom = int(width / zoom)
        new_height_zoom = int(height / zoom)
        start_x = (width - new_width_zoom) // 2
        start_y = (height - new_height_zoom) // 2
        zoomed_image = transformed_image[start_y:start_y + new_height_zoom, start_x:start_x + new_width_zoom]
        transformed_image = cv2.resize(zoomed_image, (width, height), interpolation=cv2.INTER_LINEAR)

    transformed_image = cv2.convertScaleAbs(transformed_image, alpha=contrast, beta=(brightness - 1) * 255)

    if new_width and new_height:
        transformed_image = cv2.resize(transformed_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return transformed_image


folder_level1 = get_folders(base_path)

if len(folder_level1) == 0:
    st.warning("Aucun dossier trouvé dans le chemin spécifié.")
else:
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_level1 = st.selectbox("Sélectionnez train/test/val", folder_level1)

    level1_path = os.path.join(base_path, selected_level1)
    folders_level2 = get_folders(level1_path)

    with col2:
        selected_folder = st.selectbox("Sélectionnez un dossier", folders_level2)

    folder_path = os.path.join(level1_path, selected_folder)
    image_files = get_image_files(folder_path)

    with col3:
        selected_image = st.selectbox("Sélectionnez une image", image_files)

    image_path = os.path.join(folder_path, selected_image)
    image = load_image(image_path)

    st.header("Appliquer des transformations", anchor="transform")

    transform_col_1, transform_col_2 = st.columns(2, gap="large")

    with transform_col_1:
        st.write("##### Effet visuels")

        rotate = st.slider("Rotation", -180, 180, 0)
        zoom = st.slider("Zoom", 1.0, 3.0, 1.0)
        brightness = st.slider("Luminosité", 0.5, 1.5, 1.0)
        contrast = st.slider("Contraste", 0.5, 1.5, 1.0)

        st.write("##### Taille de l'image")
        resize_col_1, resize_col_2, resize_col_3 = st.columns([2, 2, 1])

        with resize_col_1:
            new_width = st.number_input("Nouvelle largeur", min_value=1, value=image.shape[1])

        with resize_col_2:
            new_height = st.number_input("Nouvelle hauteur", min_value=1, value=image.shape[0])

        with resize_col_3:
            checkbox_64 = st.checkbox("64x64")
            checkbox_128 = st.checkbox("128x128")

        cols = st.columns(3)
        with cols[0]:
            flip_horizontal = st.checkbox("Flip Horizontal")
        with cols[1]:
            flip_vertical = st.checkbox("Flip Vertical")
        with cols[2]:
            augment_image = st.checkbox("Augment")

    if checkbox_64:
        new_width = 64
        new_height = 64

    if checkbox_128:
        new_width = 128
        new_height = 128

    transformed_image = process_image(image, rotate, zoom, brightness, contrast, new_width, new_height, flip_horizontal,
                                      flip_vertical)

    transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    with transform_col_2:
        st.image(transformed_image_rgb, use_column_width=True)

    if augment_image:
        augmented_images = augment_image_example(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY))

        cols = st.columns(len(augmented_images))

        for col, aug_img in zip(cols, augmented_images):
            col.image(aug_img, use_column_width=True, caption="Augmentation")

    original_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    transformed_array = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

    st.header("Données de l'image", anchor="data")

    original_df = pd.DataFrame(original_array.flatten(), columns=['Grayscale Value'])
    transformed_df = pd.DataFrame(transformed_array.flatten(), columns=['Grayscale Value'])

    col1, col2 = st.columns(2)

    with col1:
        st.write("Données de l'image originale")
        st.dataframe(original_df.describe().transpose())

    with col2:
        st.write("Données de l'image transformée")
        st.dataframe(transformed_df.describe().transpose())

    normalized_image = transformed_array / 255.0

    st.write("#### Histogramme des niveaux de gris de l'image")
    st.dataframe(pd.DataFrame(generate_histogram(transformed_image), columns=["Niveau de gris"]).transpose())

    st.write("#### Dataframe des niveaux de gris de chaque pixel de l'image transformée avant normalisation")
    st.dataframe(transformed_array, hide_index=False)

    st.write("#### Dataframe des pixels de l'image transformée après normalisation")
    st.dataframe(normalized_image, hide_index=False)
