import cv2
import matplotlib.pyplot as plt
from image_processing import preprocess_image, compute_difference, detect_positive_changes, detect_negative_changes, resize_image_to_match, draw_contours_on_image
from skimage.filters import threshold_otsu

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
ax1.imshow(image1, cmap='gray')
ax1.set_title('Image 1 (Référence)')

# Affichage de l'image 2 avec les contours des changements détectés
ax2.imshow(image2_with_changes, cmap='gray')
ax2.set_title(f"Image 2 avec Changements Détectés (Threshold: {initial_threshold_value:.2f})")

plt.show()
