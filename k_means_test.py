import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Base path where the files are located
base_path = "/Users/jamesdavidson/Downloads/drive-download-20240403T060912Z-001/"

# Function to load CSV files as NumPy arrays
def load_csv(file_name):
    return pd.read_csv(f"{base_path}{file_name}", header=None).to_numpy()

# Loading the datasets
electron_image = load_csv("resized_electron_image.csv")
c_ka1 = load_csv("C Ka1.csv")
si_ka1 = load_csv("Si Ka1.csv")
o_ka1 = load_csv("O Ka1.csv")
ca_ka1 = load_csv("Ca Ka1.csv")
al_ka1 = load_csv("Al Ka1.csv")
b_ka1 = load_csv("B Ka1.csv")

# Combine the arrays into a single dataset where each pixel's intensities across the maps are features
combined_data = np.stack((electron_image, c_ka1, si_ka1, o_ka1, ca_ka1, al_ka1, b_ka1), axis=-1)
combined_data_reshaped = combined_data.reshape((-1, combined_data.shape[-1]))

# Handling NaN values by replacing them with the mean of each feature column
imputer = SimpleImputer(strategy='mean')
combined_data_reshaped_imputed = imputer.fit_transform(combined_data_reshaped)

# Applying K-means clustering
kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(combined_data_reshaped_imputed)

# Reshaping the labels to the original image shape for visualization
labels_reshaped = kmeans.labels_.reshape(combined_data.shape[0], combined_data.shape[1])

# Create a mask for class 2
class_2_mask = (labels_reshaped == 2)

# Plotting both subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 10))

# Subplot 1: All classes
im = axs[0].imshow(labels_reshaped, cmap='nipy_spectral')
axs[0].set_title('All Classes')
axs[0].set_xlabel('Columns')
axs[0].set_ylabel('Rows')
axs[0].grid(False)
divider0 = make_axes_locatable(axs[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im, cax=cax0, ticks=range(1,11), label='Electron Intensity')

# Subplot 2: Electron map with Class 2 highlighted
im2 = axs[1].imshow(class_2_mask, cmap='Reds', alpha=0.5)
axs[1].set_title('Electron Map with Class 2 Highlighted in Red')
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im2, cax=cax, ticks=[0, 1], label='Electron Intensity')

plt.show()

plt.imshow(si_ka1)
plt.show()