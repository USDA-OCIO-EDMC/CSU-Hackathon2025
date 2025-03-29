import matplotlib.pyplot as plt
import rasterio
import numpy as np


class visualize:

    def plotVisual(path):
        # create raster objects from relevant files
        rastMask = rasterio.open('South_Clear_Creek/Overlap/plot.tif')
        rastHillshade = rasterio.open('South_Clear_Creek/Lidar_DEM_Hillshade/ \
                                    South_Clear_Creek_BareEarth_Hillshade_1m.tif')
        rastGrid = rasterio.open('South_Clear_Creek/Lidar_DEM_Hillshade/ \
                                South_Clear_Creek_BareEarth_DEM_1m.tif')

        # generate numpy array from rast objects (using the one and only band)
        print("Generating arrays from tif files...")
        rastMask.read(1)
        rastHillshade.read(1)
        rastGrid.read(1)

        grid = np.random.rand(10, 10)

        # Split to x (cols) and y (rows)
        rows = [coordinate[0] for coordinate in path]
        cols = [coordinate[1] for coordinate in path]

        plt.imshow(grid, cmap='gray')
        plt.plot(cols, rows, color='red', marker='o')  # (x, y) = (col, row)
        plt.title("Path over Grid")
        plt.show()