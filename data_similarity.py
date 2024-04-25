"""
DS 8013 - Deep Learning - Final Project
Ruchi Parmar - 501034872

This code checks the simialrity between data instances of both classes
"""

# Import required packages
import numpy as np
import matplotlib.pyplot as plt

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from load_data import load_data
from load_image import load_image
from pre_processing import pre_processing, pre_process_df

# load the data using
train_df, test_df = load_data()

# Function to load and preprocess images
def load_preprocess_images(image_paths, resize_dim = (224, 224)):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        if resize_dim is not None:
            image = cv2.resize(image, resize_dim, interpolation=cv2.INTER_AREA)
        images.append(image)
    return np.array(images)

# Load images from the dataframe
image_paths = train_df['image_path'].values 
train_images = load_preprocess_images(image_paths) # pre-process the images

# Flatten the images
num_images, height, width, channels = train_images.shape
flattened_images = train_images.reshape((num_images, height * width * channels))

# Standardize the data using StandardScaler library
scaler = StandardScaler()
flattened_images_standardized = scaler.fit_transform(flattened_images)

# Apply PCA to reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(flattened_images_standardized)

# Extract labels for coloring the scatter plot
labels = train_df['label_encoded'].values 

# Scatter plot of the PCA result
plt1 = plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c = labels, alpha = 0.5, cmap = 'viridis')
plt.title('PCA of Training Images')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label = 'Class Labels')
plt1.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/PCA_data.pdf")

# Apply image clustering usinf K-means
kmeans = KMeans(n_clusters = 2, random_state = 42)
cluster_labels = kmeans.fit_predict(pca_result)

# Plot the clustered data
plt2 = plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
plt.title('K-Means Clustering of Training Images (PCA-reduced Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster Label')
plt2.savefig("/Users/ruchithakor/Downloads/Masters_Docs/DS 8013 Deep Learning/Project/Skin_Cancer_TransferLearning_XAI/results/Image_clustering_data.pdf")