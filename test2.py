import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
# 1. Prétraitement des images : Application de filtres
def preprocess_image(image):
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_adaptive = cv2.fastNlMeansDenoising(image_blurred, None, 30, 7, 21)
    return image_adaptive

# 2. Calcul de la différence entre les images (log-ratio)
def compute_difference(image1, image2):
    image1_float = np.float32(image1)
    image2_float = np.float32(image2)
    difference_image = np.log1p(image2_float) - np.log1p(image1_float)
    difference_image_blurred = cv2.GaussianBlur(difference_image, (5, 5), 0)
    return  difference_image_blurred

# 3. Détection des changements positifs et négatifs
def detect_positive_changes(difference_image, threshold_value):
    binary_positive_map = difference_image > threshold_value
    kernel = np.ones((3, 3), np.uint8)
    binary_positive_map = cv2.erode(binary_positive_map.astype(np.uint8), kernel, iterations=1)
    binary_positive_map = cv2.dilate(binary_positive_map, kernel, iterations=2)
    return binary_positive_map

def detect_negative_changes(difference_image, threshold_value):
    binary_negative_map = difference_image < -threshold_value
    kernel = np.ones((3, 3), np.uint8)
    binary_negative_map = cv2.erode(binary_negative_map.astype(np.uint8), kernel, iterations=1)
    binary_negative_map = cv2.dilate(binary_negative_map, kernel, iterations=2)
    return binary_negative_map

# 4. Lire et redimensionner les images pour qu'elles soient de la même taille
def resize_image_to_match(image1, image2):
    return cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 5. Fonction pour éliminer les contours à l'aide de la transformation de distance
def remove_contours(binary_map):
    dist_transform = cv2.distanceTransform(binary_map, cv2.DIST_L2, 5)
    _, dist_thresh = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    return dist_thresh

# Fonction pour dessiner les contours sur l'image
def draw_contours_on_image(original_image, binary_map, color=(0, 255, 0), thickness=2):
    contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = original_image.copy()
    cv2.drawContours(image_with_contours, contours, -1, color, thickness)
    return image_with_contours

# Charger les images en niveaux de gris
image1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2222.png', cv2.IMREAD_GRAYSCALE)

# Redimensionner l'image2 pour qu'elle corresponde à l'image1
image2_resized = resize_image_to_match(image1, image2)

# Appliquer le prétraitement sur les images redimensionnées
image1_processed = preprocess_image(image1)
image2_processed = preprocess_image(image2_resized)

# Calculer la différence log-ratio entre les deux images
difference_image = compute_difference(image1_processed, image2_processed)

# Estimation du seuil avec Otsu
initial_threshold_value = threshold_otsu(difference_image)

# Détection des changements positifs et négatifs
binary_positive_map = detect_positive_changes(difference_image, initial_threshold_value)
binary_negative_map = detect_negative_changes(difference_image, initial_threshold_value)

# Éliminer les contours des cartes de changement
binary_positive_map_no_contours = binary_positive_map
binary_negative_map_no_contours = binary_negative_map

# Dessiner les contours des zones de changement sur image2
image2_with_changes = draw_contours_on_image(image2_resized, binary_positive_map_no_contours, color=(0, 255, 0), thickness=2)
image2_with_changes = draw_contours_on_image(image2_with_changes, binary_negative_map_no_contours, color=(255, 0, 0), thickness=2)

# Affichage des résultats dans une seule figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Affichage de l'image 1 (référence)
ax1.imshow(difference_image, cmap='gray')
ax1.set_title('Image 1 (Référence)')

# Affichage de l'image 2 avec les contours des changements détectés
ax2.imshow(image2_with_changes, cmap='gray')
ax2.set_title(f"Image 2 avec Changements Détectés (Threshold: {initial_threshold_value:.2f})")

plt.show()
