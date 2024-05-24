import cv2
import streamlit as st
import os
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

st.title("Visualisation et Prétraitement des Images de Radiographies")

base_path = os.path.join(os.path.dirname(__file__), "chest_Xray")

if not os.path.exists(base_path):
    st.error(f"Le chemin spécifié n'existe pas : {base_path}")
else:
    folder_level1 = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    if len(folder_level1) == 0:
        st.warning("Aucun dossier trouvé dans le chemin spécifié.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_level1 = st.selectbox("Sélectionnez train/test/val", folder_level1)

        level1_path = os.path.join(base_path, selected_level1)
        folders_level2 = [f for f in os.listdir(level1_path) if os.path.isdir(os.path.join(level1_path, f))]

        with col2:
            selected_folder = st.selectbox("Sélectionnez un dossier", folders_level2)

        folder_path = os.path.join(level1_path, selected_folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        with col3:
            selected_image = st.selectbox("Sélectionnez une image", image_files)

        image_path = os.path.join(folder_path, selected_image)
        image = cv2.imread(image_path)

        st.header("Appliquer des transformations")

        transform_col_1, transform_col_2 = st.columns(2, gap="large")

        with transform_col_1:
            st.write("##### Effet visuels")

            rotate = st.slider("Rotation", -180, 180, 0)
            zoom = st.slider("Zoom", 1.0, 3.0, 1.0)
            brightness = st.slider("Luminosité", 0.5, 1.5, 1.0)
            contrast = st.slider("Contraste", 0.5, 1.5, 1.0)

            st.write("##### Taille de l'image")
            resize_col_1, resize_col_2, resize_col_3 = st.columns([2,2,1])

            with resize_col_1:
                new_width = st.number_input("Nouvelle largeur", min_value=1, value=image.shape[1])

            with resize_col_2:
                new_height = st.number_input("Nouvelle hauteur", min_value=1, value=image.shape[0])

            with resize_col_3:
                checkbox_64 = st.checkbox("64x64")
                checkbox_128 = st.checkbox("128x128")

            flip_horizontal = st.checkbox("Flip Horizontal")
            flip_vertical = st.checkbox("Flip Vertical")

        transformed_image = image.copy()

        if rotate != 0:
            center = (transformed_image.shape[1] // 2, transformed_image.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)
            transformed_image = cv2.warpAffine(transformed_image, matrix, (transformed_image.shape[1], transformed_image.shape[0]))

        if flip_horizontal:
            transformed_image = cv2.flip(transformed_image, 1)

        if flip_vertical:
            transformed_image = cv2.flip(transformed_image, 0)

        if zoom != 1.0:
            height, width = transformed_image.shape[:2]
            new_width_zoom, new_height_zoom = int(width / zoom), int(height / zoom)
            resized_image_zoom = cv2.resize(transformed_image, (new_width_zoom, new_height_zoom))
            delta_w = width - new_width_zoom
            delta_h = height - new_height_zoom
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]
            transformed_image = cv2.copyMakeBorder(resized_image_zoom, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        transformed_image = cv2.convertScaleAbs(transformed_image, alpha=contrast, beta=(brightness - 1) * 255)

        if checkbox_64:
            new_width = 64
            new_height = 64

        if checkbox_128:
            new_width = 128
            new_height = 128

        transformed_image = cv2.resize(transformed_image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
        with transform_col_2:
            st.image(transformed_image_rgb, use_column_width=True)

        original_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transformed_array = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)

        st.header("Données de l'image")

        original_df = pd.DataFrame(original_array.flatten(), columns=["Valeur des Pixels"])
        transformed_df = pd.DataFrame(transformed_array.flatten(), columns=["Valeur des Pixels"])

        col1, col2 = st.columns([1, 3])

        with col1:
            st.write("Données de l'image originale")
            st.dataframe(original_df.describe())

            st.write("Données de l'image transformée")
            st.dataframe(transformed_df.describe())

        with col2:
            normalized_image = transformed_array / 255.0

            st.write("#### Dataframe des niveaux de gris de chaque pixel de l'image transformée avant normalisation")
            st.dataframe(transformed_array, hide_index=False)

            st.write("#### Dataframe des pixels de l'image transformée après normalisation")
            st.dataframe(normalized_image, hide_index=False)
