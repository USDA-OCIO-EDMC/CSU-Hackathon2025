import rasterio
import numpy as np

def load_raster(filepath, verbose=True):
    """Loads a raster file using rasterio and prints metadata."""
    with rasterio.open(filepath) as ds:
        data = ds.read()  # shape: (bands, H, W)
        if verbose:
            print(f"Loading: {filepath}")
            print(f"  Count: {ds.count}, dtype: {data.dtype}")
            print(f"  Bounds: {ds.bounds}")
            print(f"  Shape: {data.shape}")
            print(f"  NoData: {ds.nodatavals}")
    return data

def clamp_data(hillshade, roads_mask, dem, dem_range=(1, 5000), hillshade_range=(0, 65535)):
    """
    Clamps the input data. For DEM and hillshade, invalid values are set to NaN.
    For roads_mask, values outside [0, 1] are set to 0.
    """
    hillshade = hillshade.astype(np.float32)
    dem = dem.astype(np.float32)
    
    min_valid, max_valid = dem_range
    dem[(dem < min_valid) | (dem > max_valid)] = np.nan

    hmin, hmax = hillshade_range
    hillshade[hillshade < hmin] = np.nan
    hillshade[hillshade > hmax] = np.nan

    roads_mask[(roads_mask < 0) | (roads_mask > 1)] = 0

    print("\n--- AFTER CLAMPING ---")
    print(f"Hillshade: min {np.nanmin(hillshade)}, max {np.nanmax(hillshade)}, mean {np.nanmean(hillshade)}")
    print(f"Roads: min {np.nanmin(roads_mask)}, max {np.nanmax(roads_mask)}, mean {np.nanmean(roads_mask)}")
    print(f"DEM: min {np.nanmin(dem)}, max {np.nanmax(dem)}, mean {np.nanmean(dem)}")
    
    return hillshade, roads_mask, dem

def extract_patches(hillshade, dem, roads_mask, patch_size=256):
    """
    Extracts patches from the input rasters.
    Returns lists of patches for hillshade, DEM, and roads_mask.
    """
    _, H, W = hillshade.shape
    print(f"\nImage dimensions: H={H}, W={W}")
    
    hs_patches, dem_patches, rd_patches = [], [], []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            if i + patch_size <= H and j + patch_size <= W:
                hs_patches.append(hillshade[:, i:i+patch_size, j:j+patch_size])
                dem_patches.append(dem[:, i:i+patch_size, j:j+patch_size])
                rd_patches.append(roads_mask[:, i:i+patch_size, j:j+patch_size])
    
    print(f"Number of patches extracted: {len(hs_patches)}")
    return hs_patches, dem_patches, rd_patches
