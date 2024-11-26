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
def load_sentinel_image(start_date, end_date, region):
    return ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(region) \
        .filterDate(ee.Date(start_date), ee.Date(end_date)) \
        .first() \
        .clip(region)

before_change = load_sentinel_image('2015-08-01', '2015-08-31', aoi)
after_change = load_sentinel_image('2020-02-01', '2020-02-28', aoi)

# ==========================
# Fonction pour obtenir les URL des images
# ==========================
def get_image_url(image, region, vis_params, band="VV"):
    single_band_image = image.select(band)  # Select the specified band
    url = single_band_image.getThumbURL({
        'region': region,
        'dimensions': 512,
        'format': 'png',
        'min': vis_params['min'],
        'max': vis_params['max'],
        'palette': vis_params['palette'],
    })
    return url

# Paramètres de visualisation
vis_params = {
    'min': -30,
    'max': 0,
    'palette': ['black', 'white'],
}

# URL des images
before_url = get_image_url(before_change, coords, vis_params, band="VV")
after_url = get_image_url(after_change, coords, vis_params, band="VV")

# ==========================
# Chargement des images à partir des URL
# ==========================
def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        st.error(f"Erreur lors du chargement de l'image depuis l'URL : {url}")
        return None

before_image = load_image_from_url(before_url)
after_image = load_image_from_url(after_url)

# ==========================
# Affichage des images Sentinel-1
# ==========================
st.title("Visualisation des Images Radar Sentinel-1")
col1, col2 = st.columns(2)
if before_image and after_image:
  with col1:
    st.subheader("Image Avant le Changement (Août 2015)")
    st.image(before_image, caption="Image Avant", use_container_width=True)

  with col2:
    st.subheader("Image Après le Changement (Février 2020)")
    st.image(after_image, caption="Image Après", use_container_width=True)
else:
    st.error("Impossible d'afficher les images. Vérifiez votre connexion ou vos paramètres GEE.")

# ==========================
# Prétraitement des images Sentinel-1
# ==========================
filtered_image_before = preprocess_image2(before_change)  # Correction et filtrage pour avant
filtered_image_after = preprocess_image2(after_change)  # Correction et filtrage pour après


# Obtenir les URL des images prétraitées
filtered_before_url = get_image_url(filtered_image_before, coords, vis_params, band="VV")
filtered_after_url = get_image_url(filtered_image_after, coords, vis_params, band="VV")

# Charger les images prétraitées
filtered_before_image = load_image_from_url(filtered_before_url)
filtered_after_image = load_image_from_url(filtered_after_url)

# Afficher les images prétraitées
st.title("Images Prétraitées avec Correction et Filtrage")
col3, col4 = st.columns(2)
if filtered_before_image and filtered_after_image:
    with col3:
        st.subheader("Image Prétraitée Avant")
        st.image(filtered_before_image, caption="Prétraitée Avant", use_container_width=True)

    with col4:
        st.subheader("Image Prétraitée Après")
        st.image(filtered_after_image, caption="Prétraitée Après", use_container_width=True)
else:
    st.error("Impossible d'afficher les images prétraitées.")

# =========================
# Conversion depuis image radar en Opencv
# ==========================

def radar_to_opencv(image_pil):
    """
    Convertit une image radar PIL issue de Google Earth Engine en format compatible avec OpenCV.

    Parameters:
        image_pil (PIL.Image.Image): L'image radar en format PIL.

    Returns:
        np.ndarray: L'image au format compatible avec OpenCV.
    """
    # Vérifiez si l'image est en mode 'L' (grayscale)
    if image_pil.mode != 'L':
        # Convertissez en grayscale si nécessaire
        image_pil = image_pil.convert('L')
    
    # Convertissez l'image PIL en numpy array
    image_np = np.array(image_pil)

    # Normalisez les valeurs entre 0 et 255 si nécessaire
    image_cv = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return image_cv

imagecv_before_change = radar_to_opencv(filtered_before_image)
imagecv_after_change = radar_to_opencv(filtered_after_image)

cv2.imshow("Radar Image", imagecv_before_change)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ==========================
# Chargement et traitement des images utilisateur
# ==========================

def load_image(uploaded_image):
    return cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

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
    
    
