import streamlit as st

st.set_page_config(page_title="Accueil")

st.markdown("# Sommaire")

st.page_link(
    "pages/1_Visualisation_des_images.py",
    label="Visualisation des images radios",
    icon="📹",
)
