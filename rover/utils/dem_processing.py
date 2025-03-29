import os
from PIL import Image
import numpy as np

# (Optional) Silence audio warnings.
os.environ['SDL_AUDIODRIVER'] = 'dummy'
# Disable decompression bomb warnings (if you trust the image)
Image.MAX_IMAGE_PIXELS = None

def load_dem(dem_path):
    """
    Loads a DEM file using PIL and returns a numpy array.
    If the DEM has multiple bands, uses the first one.
    """
    dem_img = Image.open(dem_path)
    dem_data = np.array(dem_img)
    if dem_data.ndim > 2:
        dem_data = dem_data[0]
    return dem_data

def get_placeholder_mask(dem_data):
    """
    Computes the binary placeholder mask from DEM data.
    The mask is 1 where the cell equals the maximum value (assumed placeholder),
    and 0 elsewhere.
    """
    max_val = np.nanmax(dem_data)
    print("Maximum DEM value (assumed placeholder):", max_val)
    binary_mask = np.where(dem_data == max_val, 1, 0)
    return binary_mask
