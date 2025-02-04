import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage.morphology import binary_dilation, binary_erosion, disk
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplcursors

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

# Callback function to use with RectangleSelector
def onselect(eclick, erelease):
    global electron_image, c_ka1, si_ka1, o_ka1, ca_ka1, al_ka1, b_ka1
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    
    # Extract the ROI for all datasets
    roi_electron_image = electron_image[y1:y2, x1:x2]
    roi_c_ka1 = c_ka1[y1:y2, x1:x2]
    roi_si_ka1 = si_ka1[y1:y2, x1:x2]
    roi_o_ka1 = o_ka1[y1:y2, x1:x2]
    roi_ca_ka1 = ca_ka1[y1:y2, x1:x2]
    roi_al_ka1 = al_ka1[y1:y2, x1:x2]
    roi_b_ka1 = b_ka1[y1:y2, x1:x2]
    # Extract ROIs for o_ka1, ca_ka1, al_ka1, b_ka1 similarly
    
    # Analysis on these ROI datasets
    combined_data = np.stack((roi_electron_image, roi_c_ka1, roi_si_ka1, roi_o_ka1, roi_ca_ka1, roi_al_ka1, roi_b_ka1), axis=-1)
    combined_data_reshaped = combined_data.reshape((-1, combined_data.shape[-1]))
    
    imputer = SimpleImputer(strategy='mean')
    combined_data_reshaped_imputed = imputer.fit_transform(combined_data_reshaped)
    
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(combined_data_reshaped_imputed)
    
    labels_reshaped = kmeans.labels_.reshape(combined_data.shape[0], combined_data.shape[1])
    class_2_mask = (labels_reshaped == 2)

    class_2_mask_filtered = binary_erosion(class_2_mask, disk(3))
    class_2_mask_filtered = binary_dilation(class_2_mask_filtered, disk(6))

    # Plots within onselect
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    im = axs[0].imshow(labels_reshaped, cmap='nipy_spectral')
    axs[0].set_title('All Classes')
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax0, ticks=range(1, 11), label='Cluster')

    # To save only the left subplot as a PNG
    extent = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Adjust the file path below as necessary for your environment
    fig.savefig('/Users/jamesdavidson/Downloads/drive-download-20240403T060912Z-001/all_classes_subplot.png', bbox_inches=extent)

    # Assuming you want to save the labels_reshaped as a CSV
    labels_df = pd.DataFrame(labels_reshaped)
    csv_path = f"{base_path}kmeans_labels.csv"
    labels_df.to_csv(csv_path, index=False, header=False)

    im2 = axs[1].imshow(class_2_mask_filtered, cmap='Reds', alpha=0.5)
    axs[1].set_title('Filtered Electron Map with Class 2 Highlighted in Red')
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im2, cax=cax, ticks=[0, 1], label='Electron Intensity')
    plt.show()

# Visualize the SEM image to select the ROI
fig, ax = plt.subplots()
ax.imshow(electron_image, cmap='gray')
toggle_selector = RectangleSelector(ax, onselect, useblit=True,
                                    button=[1],  # Left mouse button
                                    minspanx=5, minspany=5,
                                    spancoords='pixels',
                                    interactive=True)
plt.show()
