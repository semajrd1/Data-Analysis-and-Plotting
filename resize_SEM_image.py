import numpy as np
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import cv2

# Load the data from CSV files
def load_csv_as_array(filepath):
    return pd.read_csv(filepath, header=None).to_numpy()

electron_image_path = '/Users/jamesdavidson/Downloads/drive-download-20240403T060912Z-001/Electron Image 1.csv'
element_map_path = '/Users/jamesdavidson/Downloads/drive-download-20240403T060912Z-001/C Ka1.csv'

# Load the images
electron_image = load_csv_as_array(electron_image_path)
element_map = load_csv_as_array(element_map_path)

# Resize the electron image to match the dimensions of the element map
electron_image_resized = resize(electron_image, element_map.shape, anti_aliasing=True)

# Normalize the resized image data to the [0, 255] range
# First, convert the image to float32 for OpenCV compatibility
electron_image_resized = electron_image_resized.astype(np.float32)
# Then, use OpenCV's normalize function to scale the pixel values
electron_image_normalized_255 = cv2.normalize(electron_image_resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Save the normalized, resized image as a PNG file
plt.imsave('/Users/jamesdavidson/Downloads/drive-download-20240403T060912Z-001/electron_image_resized_normalized.png', electron_image_normalized_255, cmap='gray')