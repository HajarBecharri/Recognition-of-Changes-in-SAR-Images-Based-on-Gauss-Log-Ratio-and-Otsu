import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skfuzzy import cmeans

# 1. Preprocessing with Gaussian Blur
def preprocess_image(image):
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    image_adaptive = cv2.fastNlMeansDenoising(image_blurred, None, 30, 7, 21)
    return image_adaptive

# 2. Gauss-Log Ratio Calculation
def compute_gauss_log_ratio(image1, image2):
    image1_float = np.float32(image1)
    image2_float = np.float32(image2)
    log_ratio = np.log1p(image2_float) - np.log1p(image1_float)  # Log-ratio calculation
    gauss_log_ratio = cv2.GaussianBlur(log_ratio, (5, 5), 0)     # Apply Gaussian smoothing
    return gauss_log_ratio

# 3. FCM-Based Classification (Part of MRFFCM)
def fuzzy_c_means_clustering(image, num_clusters=2):
    # Flatten the image for clustering
    data = image.flatten()
    data = data[np.isfinite(data)]  # Remove NaN values (if any)
    data = data.reshape(1, -1)
    
    # Fuzzy C-Means Clustering
    cntr, u, _, _, _, _, _ = cmeans(data, num_clusters, 2, error=0.005, maxiter=1000)
    cluster_labels = np.argmax(u, axis=0)  # Get the cluster labels
    
    # Reshape the cluster labels back to the original image shape
    classified_image = np.zeros(image.shape, dtype=np.uint8)
    classified_image[np.isfinite(image)] = cluster_labels
    return classified_image

# 4. Apply MRFFCM (Spatial Smoothing + FCM)
def apply_mrffcm(image):
    # Use FCM for initial clustering
    fcm_clusters = fuzzy_c_means_clustering(image, num_clusters=2)
    
    # Apply spatial smoothing (optional, can be part of the MRF step)
    kernel = np.ones((3, 3), np.uint8)
    smoothed_clusters = cv2.morphologyEx(fcm_clusters, cv2.MORPH_CLOSE, kernel)
    return smoothed_clusters

# 5. Read and preprocess images
image1 = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2222.png', cv2.IMREAD_GRAYSCALE)
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
image1_processed = preprocess_image(image1)
image2_processed = preprocess_image(image2_resized)

# 6. Compute Gauss-Log Ratio
difference_image = compute_gauss_log_ratio(image1_processed, image2_processed)

# 7. Apply MRFFCM for change detection
change_map = apply_mrffcm(difference_image)

# 8. Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(image1, cmap='gray')
ax1.set_title('Image 1 (Reference)')

ax2.imshow(image2_resized, cmap='gray')
ax2.set_title('Image 2 (Current)')

ax3.imshow(change_map, cmap='gray')
ax3.set_title('Detected Changes (MRFFCM)')

plt.show()
