import streamlit as st
import cv2
import numpy as np
from image_processing import process_images

def load_image(uploaded_image):
    return cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

# Interface Streamlit
st.title("🚀 Traitement d'Image pour Détection de Changement")

# Chargement des images
uploaded_image1 = st.file_uploader("Téléchargez la première image", type=["jpg", "png", "jpeg"])
uploaded_image2 = st.file_uploader("Téléchargez la deuxième image", type=["jpg", "png", "jpeg"])

if uploaded_image1 is not None and uploaded_image2 is not None:
    # Lire les images téléchargées
    image1 = load_image(uploaded_image1)
    image2 = load_image(uploaded_image2)
    
    # Appliquer le traitement
    image1, image2_resized, difference_image, binary_positive_map, binary_negative_map, image2_with_changes, threshold_value = process_images(image1, image2)
    
    # Clamer ou normaliser les images avant de les afficher
    image1_clamped = np.clip(image1, 0, 255).astype(np.uint8)
    image2_resized_clamped = np.clip(image2_resized, 0, 255).astype(np.uint8)
    
    # Pour les cartes binaires, nous les convertissons en uint8 (0 ou 255)
    binary_positive_map_clamped = (binary_positive_map.astype(np.uint8)) * 255
    binary_negative_map_clamped = (binary_negative_map.astype(np.uint8)) * 255
    
    # Normalisation de l'image de différence pour l'affichage
    difference_image_normalized = cv2.normalize(difference_image, None, 0, 255, cv2.NORM_MINMAX)
    difference_image_normalized = difference_image_normalized.astype(np.uint8)
    
    # Pour l'image de résultats avec contours
    image2_with_changes_clamped = np.clip(image2_with_changes, 0, 255).astype(np.uint8)

    # Affichage des images avec un titre
    st.markdown('<h2 style="color:#4CAF50;">Images d\'Origine</h2>', unsafe_allow_html=True)
    st.write("### Voici les deux images d'origine avant de commencer la détection des changements.")
    
    # Affichage des images côte à côte
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1_clamped, caption="Image 1 (Origine)", channels="GRAY", use_container_width=True)
    with col2:
        st.image(image2_resized_clamped, caption="Image 2 (Origine)", channels="GRAY", use_container_width=True)
    
    # Grand titre pour la détection
    st.markdown('<h1 style="color:#FF5722;">Commençons la Détection 🔍</h1>', unsafe_allow_html=True)

    # Affichage des étapes de la détection avec des titres et descriptions
    st.markdown('<h3 style="color:#2196F3;">Différence entre Image 1 et Image 2</h3>', unsafe_allow_html=True)
    st.write("Nous calculons la différence entre les deux images en utilisant une approche logarithmique pour mieux détecter les changements.")
    st.image(difference_image_normalized, caption="Différence (logarithmique)", channels="GRAY", use_container_width=True)
    
    st.markdown('<h3 style="color:#8BC34A;">Carte des Changements Positifs</h3>', unsafe_allow_html=True)
    st.write(f"Cette carte montre les régions de l'image où des changements positifs ont été détectés, selon un seuil basé sur l'otsu. (Seuil = {threshold_value:.2f})")
    st.image(binary_positive_map_clamped, caption="Changements Positifs", channels="GRAY", use_container_width=True)
    
    st.markdown('<h3 style="color:#FFC107;">Carte des Changements Négatifs</h3>', unsafe_allow_html=True)
    st.write(f"Cette carte montre les régions de l'image où des changements négatifs ont été détectés. (Seuil = {threshold_value:.2f})")
    st.image(binary_negative_map_clamped, caption="Changements Négatifs", channels="GRAY", use_container_width=True)
    
    st.markdown('<h3 style="color:#9C27B0;">Image 2 avec Contours des Changements</h3>', unsafe_allow_html=True)
    st.write("Voici l'image 2 avec les contours des régions où des changements positifs et négatifs ont été détectés. Les contours sont colorés pour les différencier.")
    st.image(image2_with_changes_clamped, caption="Image avec Contours", channels="GRAY", use_container_width=True)
    
    
