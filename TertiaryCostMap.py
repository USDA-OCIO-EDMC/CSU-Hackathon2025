import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import zoom

# This class generates a Teritiary Cost Map which is an array of coordinate
# values that are limited to the following values with the following
# representations
# 0: transversible
# 1: obstacle
# 2: path (also identified as transversible)

# These data is generated from the a base of the mask.tif that serves as a
# boundary and an overlay / intersection of coordinates
# from the BareEarth DEM 1m tif file


class CostMap:
    @staticmethod
    def getTCM():
        # Open raster files
        rastMask = rasterio.open('South_Clear_Creek/Overlap/South_Clear_Creek_Roads_Mask.tif')
        rastGrid = rasterio.open('South_Clear_Creek/Lidar_DEM_Hillshade/South_Clear_Creek_BareEarth_DEM_1m.tif')

        # Generate numpy arrays from raster files (using the first band)
        print("Generating arrays from tif files...")
        maskArray = rastMask.read(1)  # Mask array indicating traversable areas
        gridArray = rastGrid.read(1)  # DEM data

        rows, cols = gridArray.shape
        transform = rastGrid.transform

        # Generate coordinates based on raster transform (downsampling every 10th pixel)
        coords = []
        for row in range(0, rows, 10):
            for col in range(0, cols, 10):
                x, y = xy(transform, col, row)
                coords.append((x, y))

        grid_tert = np.zeros_like(gridArray, dtype=np.uint8)

        print("Overriding untraversable areas with paths...")
        maskArray_resized = zoom(maskArray,
                                 (grid_tert.shape[0] / maskArray.shape[0],
                                  grid_tert.shape[1] / maskArray.shape[1]),
                                 order=0)

        print(f"Original maskArray shape: {maskArray.shape}")
        print(f"Resized maskArray shape: {maskArray_resized.shape}")
        print(f"grid_tert shape: {grid_tert.shape}")

        # Ensure the resized maskArray has the exact same shape as grid_tert
        if maskArray_resized.shape != grid_tert.shape:
            print("Resizing failed to match the shape. Adjusting manually.")
            maskArray_resized = zoom(maskArray,
                                     (grid_tert.shape[0] / maskArray.shape[0],
                                      grid_tert.shape[1] / maskArray.shape[1]),
                                     order=0)

        grid_tert[maskArray_resized == 1] = 0  # 0 is untraversable (or paths)

        print("Adding random obstacles...")
        num_obstacles = 10
        flat_indices = np.random.choice(rows * cols,
                                        size=num_obstacles,
                                        replace=False)
        for idx in flat_indices:
            i = idx // cols
            j = idx % cols
            grid_tert[i, j] = 1  # 1 represents an obstacle

        print(f"Generated Cost Map with shape: {grid_tert.shape}")
        return grid_tert
