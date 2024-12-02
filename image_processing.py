import numpy as np
import cv2
from skimage.filters import threshold_otsu
import ee
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import math

def to_natural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def to_db(img):
    return ee.Image(img).log10().multiply(10.0)

def terrain_correction(image):
    img_geom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(img_geom)
    sigma0_pow = ee.Image.constant(10).pow(image.divide(10.0))

    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(
        ee.Reducer.mean(),
        theta_i.get('system:footprint'),
        1000
    ).get('aspect')

    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')

    # Ensure phi_r is properly initialized as a constant image if needed
    phi_r = ee.Image.constant(0).subtract(phi_s)  # Update this part if needed
    phi_r_rad = phi_r.multiply(ee.Image.constant(math.pi / 180))
    alpha_s_rad = alpha_s.multiply(ee.Image.constant(math.pi / 180))
    theta_i_rad = theta_i.multiply(ee.Image.constant(math.pi / 180))
    ninety_rad = ee.Image.constant(90).multiply(ee.Image.constant(math.pi / 180))

    alpha_r = (alpha_s_rad.tan().multiply(phi_r_rad.cos())).atan()
    alpha_az = (alpha_s_rad.tan().multiply(phi_r_rad.sin())).atan()

    theta_lia = (alpha_az.cos().multiply((theta_i_rad.subtract(alpha_r)).cos())).acos()
    theta_lia_deg = theta_lia.multiply(180 / math.pi)

    gamma0 = sigma0_pow.divide(theta_i_rad.cos())
    gamma0_db = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0_db.select('VV').subtract(gamma0_db.select('VH'))

    nominator = (ninety_rad.subtract(theta_i_rad).add(alpha_r)).tan()
    denominator = (ninety_rad.subtract(theta_i_rad)).tan()
    vol_model = (nominator.divide(denominator)).abs()

    gamma0_volume = gamma0.divide(vol_model)
    gamma0_volume_db = ee.Image.constant(10).multiply(gamma0_volume.log10())

    alpha_r_deg = alpha_r.multiply(180 / math.pi)
    layover = alpha_r_deg.lt(theta_i)
    shadow = theta_lia_deg.lt(85)

    ratio = gamma0_volume_db.select('VV').subtract(gamma0_volume_db.select('VH'))

    output = gamma0_volume_db.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_i_rad)
    output = output.addBands(layover).addBands(shadow).addBands(gamma0_db).addBands(ratio_1)

    return image.addBands(
        output.select(['VV', 'VH'], ['VV', 'VH']),
        None,
        True
    )


def RefinedLee(img):
    # img doit être en unités naturelles, pas en dB !
    myimg = to_natural(img)  # Assurez-vous d'utiliser la fonction de conversion appropriée

    # Configuration des noyaux 3x3
    weights3 = ee.List.repeat(ee.List.repeat(1, 3), 3)
    kernel3 = ee.Kernel.fixed(3, 3, weights3, 1, 1, False)

    # Calculer la moyenne et la variance en utilisant le noyau 3x3
    mean3 = myimg.reduceNeighborhood(ee.Reducer.mean(), kernel3)
    variance3 = myimg.reduceNeighborhood(ee.Reducer.variance(), kernel3)

    # Calcul des statistiques et directions pour le filtrage
    # Définir dir_var, dir_mean, et sigmaV ici
    dir_var = variance3
    dir_mean = mean3
    sigmaV = dir_var.sqrt() 

    # Générer la valeur filtrée
    varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))
    b = varX.divide(dir_var)
    result = dir_mean.add(b.multiply(myimg.subtract(dir_mean)))

    # Retourner l'image filtrée
    return img.addBands(result.rename("filter"))

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Erreur lors du chargement de l'image depuis l'URL : {url}")
        return None
    
    
def get_image_url(image, region, band="VV"):
    single_band_image = image.select(band)  # Select the specified band
    url = single_band_image.getThumbURL({
        'region': region,
        'dimensions': 512,
        'format': 'png',
        'min': -30,
        'max': 0,
        'palette': ['black', 'white'],
    })
    return url

def preprocess_image2(image):
    image_filterd = terrain_correction(image)
    image_filterd2 = RefinedLee(image_filterd)

    return image_filterd2


def radar_to_opencv(image_pil):
    if image_pil.mode != 'L':
        image_pil = image_pil.convert('L')
    image_np = np.array(image_pil)
    image_cv = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image_cv

def preprocess_image(image):
    """Applique un prétraitement d'image avec un flou et un débruitage"""
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_adaptive = cv2.fastNlMeansDenoising(image_blurred, None, 30, 7, 21)
    print("Hello procees images ")
    return image_adaptive

def compute_difference(image1, image2):
    """Calcule la différence entre deux images"""
    image1_float = np.float32(image1)
    image2_float = np.float32(image2)
    difference_image = np.log1p(image2_float) - np.log1p(image1_float)
    difference_image_blurred = cv2.GaussianBlur(difference_image, (5, 5), 0)
    return difference_image_blurred

def detect_positive_changes(difference_image, threshold_value):
    """Détecte les changements positifs dans l'image de différence"""
    binary_positive_map = difference_image > threshold_value
    kernel = np.ones((3, 3), np.uint8)
    binary_positive_map = cv2.erode(binary_positive_map.astype(np.uint8), kernel, iterations=1)
    binary_positive_map = cv2.dilate(binary_positive_map, kernel, iterations=2)
    return binary_positive_map

def detect_negative_changes(difference_image, threshold_value):
    """Détecte les changements négatifs dans l'image de différence"""
    binary_negative_map = difference_image < -threshold_value
    kernel = np.ones((3, 3), np.uint8)
    binary_negative_map = cv2.erode(binary_negative_map.astype(np.uint8), kernel, iterations=1)
    binary_negative_map = cv2.dilate(binary_negative_map, kernel, iterations=2)
    return binary_negative_map

def resize_image_to_match(image1, image2):
    """Redimensionne l'image 2 pour qu'elle ait la même taille que l'image 1"""
    return cv2.resize(image2, (image1.shape[1], image1.shape[0]))

def draw_contours_on_image(original_image, binary_map, color=(0, 255, 0), thickness=2):
    """Dessine les contours des changements sur l'image originale"""
    contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = original_image.copy()
    cv2.drawContours(image_with_contours, contours, -1, color, thickness)
    return image_with_contours

def process_images(image1, image2 , coords):
    """Processus complet des images : prétraitement, calcul de la différence, détection des changements"""

    filtered_image_before = preprocess_image2(image1)  # Correction et filtrage pour avant
    filtered_image_after = preprocess_image2(image2)  # Correction et filtrage pour après
    # URL des images
    before_url = get_image_url(image1, coords, band="VV")
    after_url = get_image_url(image2, coords, band="VV")

    before_image = load_image_from_url(before_url)
    after_image = load_image_from_url(after_url)

    image1_original = radar_to_opencv(before_image)
    image2_original = radar_to_opencv(after_image)

    # Obtenir les URL des images prétraitées
    filtered_before_url = get_image_url(filtered_image_before, coords, band="VV")
    filtered_after_url = get_image_url(filtered_image_after, coords, band="VV")

    # Charger les images prétraitées
    filtered_before_image = load_image_from_url(filtered_before_url)
    filtered_after_image = load_image_from_url(filtered_after_url)

    imagecv_before_change = radar_to_opencv(filtered_before_image)
    imagecv_after_change = radar_to_opencv(filtered_after_image)

    image1_filtered = imagecv_before_change
    image2_filtered = imagecv_after_change

    image1 = image1_filtered
    image2 = image2_filtered

    # Redimensionner image 2 pour qu'elle corresponde à la taille d'image 1
    image2_resized = resize_image_to_match(image1, image2)
    
    # Appliquer le prétraitement sur les images
    image1_processed = preprocess_image(image1)
    image2_processed = preprocess_image(image2_resized)
    
    # Calculer la différence entre les deux images
    difference_image = compute_difference(image1_processed, image2_processed)
    
    # Appliquer un seuil pour détecter les changements
    threshold_value = threshold_otsu(difference_image)
    
    # Détecter les changements positifs et négatifs
    binary_positive_map = detect_positive_changes(difference_image, threshold_value)
    binary_negative_map = detect_negative_changes(difference_image, threshold_value)
    
    # Dessiner les contours des changements sur image 2
    image2_with_changes = draw_contours_on_image(image2_resized, binary_positive_map, color=(0, 255, 0), thickness=2)
    image2_with_changes = draw_contours_on_image(image2_with_changes, binary_negative_map, color=(255, 0, 0), thickness=2)
    
    return image1_original, image2_original,image1_filtered , image2_filtered , difference_image, binary_positive_map, binary_negative_map, image2_with_changes, threshold_value
