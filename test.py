

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger les images satellites T1 et T2
image_t1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image_t2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

image_t2 = cv2.resize(image_t2, (image_t1.shape[1], image_t1.shape[0]))
# Vérifier que les images sont bien chargées
if image_t1 is None or image_t2 is None:
    raise ValueError("Les images n'ont pas été chargées correctement.")

# Eviter la division par zéro
image_t2 = np.maximum(image_t2, 1e-10)  # Ajouter une petite valeur pour éviter les divisions par zéro

# Calcul du log-ratio
log_ratio = np.log(image_t2 / image_t1)

# Seuillage pour détecter les changements
_, change_map = cv2.threshold(log_ratio, 0.2, 255, cv2.THRESH_BINARY)

# Affichage des résultats
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Image T1")
plt.imshow(image_t1, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Image T2")
plt.imshow(image_t2, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Carte des Changements")
plt.imshow(change_map, cmap='hot')
plt.axis('off')

plt.tight_layout()
plt.show()
