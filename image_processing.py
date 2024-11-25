import numpy as np
import cv2
from skimage.filters import threshold_otsu

def preprocess_image(image):
    """Applique un prétraitement d'image avec un flou et un débruitage"""
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_adaptive = cv2.fastNlMeansDenoising(image_blurred, None, 30, 7, 21)
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

def process_images(image1, image2):
    """Processus complet des images : prétraitement, calcul de la différence, détection des changements"""
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
    
    return image1, image2_resized, difference_image, binary_positive_map, binary_negative_map, image2_with_changes, threshold_value
