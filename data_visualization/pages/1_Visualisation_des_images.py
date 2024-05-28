import streamlit as st
import os
from PIL import Image

st.set_page_config(
    layout="wide",
)

st.title("Visualisation des images radios")

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))

if not os.path.exists(base_path):
    st.error(f"Le chemin spécifié n'existe pas : {base_path}")
else:
    folder_level1 = [
        f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))
    ]

    if len(folder_level1) == 0:
        st.warning("Aucun dossier trouvé dans le chemin spécifié.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            selected_level1 = st.selectbox("Sélectionnez train/test/val", folder_level1)

        level1_path = os.path.join(base_path, selected_level1)
        folders_level2 = [
            f
            for f in os.listdir(level1_path)
            if os.path.isdir(os.path.join(level1_path, f))
        ]

        with col2:
            selected_folder = st.selectbox("Sélectionnez un dossier", folders_level2)

        folder_path = os.path.join(level1_path, selected_folder)
        image_files = [
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

        images_per_page = 3
        total_images = len(image_files)
        total_pages = (total_images - 1) // images_per_page + 1

        page_number = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1
        )

        start_index = (page_number - 1) * images_per_page
        end_index = start_index + images_per_page

        st.write(
            f"Images dans le dossier {selected_folder} (page {page_number}/{total_pages}):"
        )

        cols = st.columns(images_per_page)
        for i, image_file in enumerate(image_files[start_index:end_index]):
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path)
            cols[i].image(image, caption=image_file)
