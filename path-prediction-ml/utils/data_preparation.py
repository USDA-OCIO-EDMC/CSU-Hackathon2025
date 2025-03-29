import numpy as np
from utils.data_io import load_raster, clamp_data, extract_patches
from utils.augmentation import get_augmentation_transform
from utils.dataset_module import oversample_until_balanced, create_augmented_dataset, split_dataset
from torch.utils.data import Subset

def prepare_data(subset_size=None, oversample=True):
    # File paths.
    hillshade_path = "./South_Clear_Creek/Lidar_DEM_Hillshade/South_Clear_Creek_BareEarth_Hillshade_1m.tif"
    roads_path = "./South_Clear_Creek/Roads_Boundary/South_Clear_Creek_Roads_Mask.tif"
    dem_path = "./South_Clear_Creek/Lidar_DEM_Hillshade/South_Clear_Creek_BareEarth_DEM_1m.tif"
    
    hillshade = load_raster(hillshade_path)
    roads_mask = load_raster(roads_path)
    dem = load_raster(dem_path)
    
    print("\n--- RAW DATA STATS ---")
    print(f"Hillshade -> min: {np.nanmin(hillshade)}, max: {np.nanmax(hillshade)}, mean: {np.nanmean(hillshade)}")
    print(f"Roads -> min: {roads_mask.min()}, max: {roads_mask.max()}, mean: {roads_mask.mean()}")
    print(f"DEM -> min: {np.nanmin(dem)}, max: {np.nanmax(dem)}, mean: {np.nanmean(dem)}")
    
    hillshade, roads_mask, dem = clamp_data(hillshade, roads_mask, dem)
    hs_patches, dem_patches, rd_patches = extract_patches(hillshade, dem, roads_mask, patch_size=256)
    
    print(f"Dataset size before oversampling: {len(hs_patches)} patches")
    
    if oversample:
        hs_patches, dem_patches, rd_patches = oversample_until_balanced(hs_patches, dem_patches, rd_patches,
                                                                         minority_threshold=0.01)
    
    augmentation_transform = get_augmentation_transform()
    full_dataset = create_augmented_dataset(hs_patches, dem_patches, rd_patches, augmentation_transform)
    print(f"Total dataset size after augmentation: {len(full_dataset)} samples")
    
    if subset_size is not None:
        print(f"\nUsing a subset of the data: {subset_size} samples")
        indices = list(range(min(subset_size, len(full_dataset))))
        full_dataset = Subset(full_dataset, indices)
    
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset,
                                                              train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    return train_dataset, val_dataset, test_dataset
