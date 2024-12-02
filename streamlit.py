import streamlit as st
import cv2
import numpy as np
from image_processing import process_images
from image_processing import preprocess_image2
from image_processing import RefinedLee
from image_processing import terrain_correction
import ee
import requests
from PIL import Image
from io import BytesIO

# ==========================
# Initialisation
# ==========================
st.title("🚀 Traitement d'Image pour Détection de Changement")

# Authentification et initialisation de Google Earth Engine
ee.Authenticate()
ee.Initialize()

# ==========================
# Définition de la région d'intérêt (AOI)
# ==========================
geoJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.473892211914062, 49.98081240937428],
                        [8.658599853515625, 49.98081240937428],
                        [8.658599853515625, 50.06066538593667],
                        [8.473892211914062, 50.06066538593667],
                        [8.473892211914062, 49.98081240937428],
                    ]
                ],
            },
        }
    ],
}

# Création d'une géométrie pour AOI
coords = geoJSON['features'][0]['geometry']['coordinates']
aoi = ee.Geometry.Polygon(coords)

# ==========================
# Chargement des images radar Sentinel-1
# ==========================

image_before_change = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filterDate(ee.Date('2015-08-01'), ee.Date('2015-08-31')) \
        .first() \
        .clip(aoi)

image_after_change = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filterDate(ee.Date('2020-02-01'), ee.Date('2020-02-28')) \
        .first() \
        .clip(aoi)

# ==========================
# traitement des images 
# ==========================
    
# Appliquer le traitement
image1, image2_resized, image_before_change_filtered , image_after_change_filtered ,difference_image, binary_positive_map, binary_negative_map, image2_with_changes, threshold_value = process_images(image_before_change, image_after_change , coords)

# ==========================
# Affichage des images Sentinel-1
# ==========================
st.title("Visualisation des Images Radar Sentinel-1")
col1, col2 = st.columns(2)
with col1:
 st.subheader("Image Avant le Changement (Août 2015)")
 st.image(image1, caption="Image Avant", use_container_width=True)

with col2:
 st.subheader("Image Après le Changement (Février 2020)")
 st.image(image2_resized, caption="Image Après", use_container_width=True)

# ==========================
# Afficher les images prétraitées
# ==========================
st.title("Images Prétraitées avec Correction et Filtrage")
col3, col4 = st.columns(2)
with col3:
    st.subheader("Image Prétraitée Avant")
    st.image(image_before_change_filtered, caption="Prétraitée Avant", use_container_width=True)

with col4:
    st.subheader("Image Prétraitée Après")
    st.image(image_after_change_filtered, caption="Prétraitée Après", use_container_width=True)

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
    
    
