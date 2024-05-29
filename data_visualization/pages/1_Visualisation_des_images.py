import sys

import numpy as np
import pandas as pd
import streamlit as st
import os
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from skimage.feature import graycomatrix, graycoprops

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.preprocessing import load_images_from_folder  # noqa: E402

st.set_page_config(
    layout="wide",
)


@st.cache_data
def load_images(folder_path):
    return load_images_from_folder(folder_path)


@st.cache_data
def generate_pca(X_train):
    X_flattened = X_train.reshape(X_train.shape[0], -1)
    pca = PCA(n_components=2)
    return pca.fit_transform(X_flattened)


@st.cache_data
def get_corr_mtx(X_train):
    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    all_features = []
    for image in X_train:
        features = extract_glcm_features(image, distances, angles)
        all_features.append(features)

    features_df = pd.DataFrame(
        all_features,
        columns=[
            "contrast_1_0",
            "contrast_1_45",
            "contrast_1_90",
            "contrast_1_135",
            "contrast_2_0",
            "contrast_2_45",
            "contrast_2_90",
            "contrast_2_135",
            "dissimilarity_1_0",
            "dissimilarity_1_45",
            "dissimilarity_1_90",
            "dissimilarity_1_135",
            "dissimilarity_2_0",
            "dissimilarity_2_45",
            "dissimilarity_2_90",
            "dissimilarity_2_135",
            "homogeneity_1_0",
            "homogeneity_1_45",
            "homogeneity_1_90",
            "homogeneity_1_135",
            "homogeneity_2_0",
            "homogeneity_2_45",
            "homogeneity_2_90",
            "homogeneity_2_135",
            "energy_1_0",
            "energy_1_45",
            "energy_1_90",
            "energy_1_135",
            "energy_2_0",
            "energy_2_45",
            "energy_2_90",
            "energy_2_135",
            "correlation_1_0",
            "correlation_1_45",
            "correlation_1_90",
            "correlation_1_135",
            "correlation_2_0",
            "correlation_2_45",
            "correlation_2_90",
            "correlation_2_135",
            "ASM_1_0",
            "ASM_1_45",
            "ASM_1_90",
            "ASM_1_135",
            "ASM_2_0",
            "ASM_2_45",
            "ASM_2_90",
            "ASM_2_135",
        ],
    )

    return features_df.corr()


def extract_glcm_features(image, distances, angles):
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = []
    for prop in [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]:
        feature = graycoprops(glcm, prop)
        features.append(feature.flatten())
    return np.concatenate(features)


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
train_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/raw/train")
)
test_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/raw/test")
)

X_train, y_train = load_images(train_path)
X_test, y_test = load_images(test_path)

train_normal_class, train_pneumonia_class = (
    np.where(y_train == 0)[0],
    np.where(y_train == 1)[0],
)

X_train_normal, y_train_normal = (
    X_train[train_normal_class],
    y_train[train_normal_class],
)
X_train_pneumonia, y_train_pneumonia = (
    X_train[train_pneumonia_class],
    y_train[train_pneumonia_class],
)

if not os.path.exists(base_path):
    st.error(f"Le chemin spécifié n'existe pas : {base_path}")
else:
    folder_level1 = [
        f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))
    ]

    if len(folder_level1) == 0:
        st.warning("Aucun dossier trouvé dans le chemin spécifié.")
    else:
        st.write("#### Visualisation des images")

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

    st.write("#### Distribution des classes")
    plot_col_1, plot_col_2 = st.columns(2)

    with plot_col_1:
        st.write("##### Données d'entraînement")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_train)
        plt.xlabel("Classe")
        plt.ylabel("Nombre d'images")
        st.pyplot(plt)

    with plot_col_2:
        st.write("##### Données de test")
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y_test)
        plt.xlabel("Classe")
        plt.ylabel("Nombre d'images")
        st.pyplot(plt)

    st.write("#### Images moyennes")
    mean_img_col_1, mean_img_col_2 = st.columns(2)

    with mean_img_col_1:
        st.write("##### Image moyenne d'un patient sain")
        mean_img_normal = np.mean(X_train_normal, axis=0) / 255
        st.image(mean_img_normal, use_column_width=True)

    with mean_img_col_2:
        st.write("##### Image moyenne d'un patient malade")
        mean_img_pneumonia = np.mean(X_train_pneumonia, axis=0) / 255
        st.image(mean_img_pneumonia, use_column_width=True)

    col_pca, col_diff, col_corr = st.columns(3)

    with col_pca:
        st.write("##### PCA")
        X_pca = generate_pca(X_train)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="viridis")
        plt.xlabel("Pas malade")
        plt.ylabel("Malade")
        st.pyplot(plt)

    with col_diff:
        st.write("##### Différences Sain/Malade ")
        contrast_mean = (np.mean(X_train_normal, axis=0) / 255) - (
            np.mean(X_train_pneumonia, axis=0) / 255
        )
        plt.imshow(contrast_mean, cmap="bwr")
        st.pyplot(plt)

    with col_corr:
        st.write("##### Matrice de Corrélation des Caractéristiques GLCM")
        plt.figure(figsize=(12, 10))
        sns.heatmap(get_corr_mtx(X_train), annot=False, cmap="coolwarm")
        st.pyplot(plt)
