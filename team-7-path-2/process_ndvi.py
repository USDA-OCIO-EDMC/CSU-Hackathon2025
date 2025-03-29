import rasterio
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from rasterio.warp import reproject


def select_bands(src):
    # Read the NIR (band 4) and Red (band 1) bands
    nir_band = np.where(np.isnan(src.read(4).astype(float)),
                        np.inf, src.read(4).astype(float))
    red_band = np.where(np.isnan(src.read(1).astype(float)),
                        np.inf, src.read(1).astype(float))
    return nir_band, red_band


def get_grid_mean(grid):
    return np.mean(grid)


def break_into_grids(ndvi_data, grid_size, transform_matrix, src):
    height, width = ndvi_data.shape

    num_grids_height = int(np.ceil(height / grid_size))
    num_grids_width = int(np.ceil(width / grid_size))

    grids = np.zeros((num_grids_height, num_grids_width))

    for i in range(num_grids_height):
        for j in range(num_grids_width):
            x_start = j * grid_size
            x_end = min((j + 1) * grid_size, width)
            y_start = i * grid_size
            y_end = min((i + 1) * grid_size, height)

            grid = ndvi_data[y_start:y_end, x_start:x_end]

            grid_mean = np.mean(grid)

            grids[i, j] = grid_mean

    return grids


def normalize_image(image):
    # Normalize the pixel values to a common range (e.g., 0-1)
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image


def apply_filter(image):
    # Apply a median filter or a Gaussian filter to reduce noise and smooth out the image
    filtered_image = filters.median(image)
    # filtered_image = filters.gaussian(image, sigma=1)
    return filtered_image


def calculate_ndvi(nir_band, red_band):
    ndvi = np.where(np.isnan(nir_band) | np.isnan(red_band),
                    np.inf, (nir_band - red_band) / (nir_band + red_band))
    print(ndvi[ndvi != 0])
    return ndvi


def preprocess_data(file_path):
    # Open the TIFF file
    with rasterio.open(file_path) as src:
        print("TIFF file opened successfully.")

        # Select the NIR and Red bands
        nir_band, red_band = select_bands(src)

        print("NIR and Red bands selected successfully.")

        # Print the shape and type of the bands
        print("NIR band shape:", nir_band.shape)
        print("NIR band type:", nir_band.dtype)
        print("Red band shape:", red_band.shape)
        print("Red band type:", red_band.dtype)

        # Normalize the images
        nir_band_normalized = normalize_image(nir_band)
        red_band_normalized = normalize_image(red_band)
        print("Images normalized successfully.")

        # Print the minimum and maximum values of the normalized bands
        print("NIR band normalized min:", np.min(nir_band_normalized))
        print("NIR band normalized max:", np.max(nir_band_normalized))
        print("Red band normalized min:", np.min(red_band_normalized))
        print("Red band normalized max:", np.max(red_band_normalized))

        # Apply a filter to the images
        nir_band_filtered = apply_filter(nir_band_normalized)
        red_band_filtered = apply_filter(red_band_normalized)
        print("Images filtered successfully.")

        # Print the minimum and maximum values of the filtered bands
        print("NIR band filtered min:", np.min(nir_band_filtered))
        print("NIR band filtered max:", np.max(nir_band_filtered))
        print("Red band filtered min:", np.min(red_band_filtered))
        print("Red band filtered max:", np.max(red_band_filtered))

        # Visualize the filtered bands
        print("calculating ndvi...")
        ndvi = calculate_ndvi(nir_band_filtered, red_band_filtered)
        grid_size = 1
        print("breaking into grids...")

        grids = break_into_grids(
            ndvi, grid_size, src.transform, src)

        print(f'width = {len(grids[0])}')
        print(f'height = {len(grids)}')

        np.save('ndvi_points_no_null.npy', grids)

    return nir_band_filtered, red_band_filtered


# Example usage
file_path = 'South_Clear_Creek_2023_NAIP_1m.tif'
nir_band_filtered, red_band_filtered = preprocess_data(file_path)
