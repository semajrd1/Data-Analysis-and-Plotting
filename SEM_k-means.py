import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Base path where the files are located
base_path = "/Users/jamesdavidson/Downloads/drive-download-20240403T060912Z-001/"

# Function to load CSV files as NumPy arrays
def load_csv(file_name):
    return pd.read_csv(f"{base_path}{file_name}", header=None).to_numpy()

# Loading the electron image
electron_image = load_csv("resized_electron_image.csv")

# Reshape the electron image for K-means clustering
electron_image_reshaped = electron_image.reshape((-1, 1))

# Handling NaN values by replacing them with the mean of the electron image
imputer = SimpleImputer(strategy='mean')
electron_image_reshaped_imputed = imputer.fit_transform(electron_image_reshaped)

# Applying K-means clustering
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(electron_image_reshaped_imputed)

# Reshaping the labels to the original image shape for visualization
labels_reshaped = kmeans.labels_.reshape(electron_image.shape)

# Visualizing the clustering result
plt.figure(figsize=(10, 10))
plt.imshow(labels_reshaped, cmap='nipy_spectral')
plt.colorbar(label='Cluster')
plt.title('K-Means Clustering Result for Electron Image')
plt.show()

# Plotting cluster 2 on top of the electron image
cluster_2_mask = (labels_reshaped == 2)
plt.imshow(cluster_2_mask, cmap='Reds', alpha=0.5)  # Overlay cluster 2 in red on top of the electron image

plt.show()