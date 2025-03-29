import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split

def normalize_channel(x):
    """Replace NaNs with 0 and min–max normalize a single-channel np.array."""
    x = np.nan_to_num(x, nan=0)
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val - min_val > 1e-10:
        return (x - min_val) / (max_val - min_val)
    return x

class CreekDataset(Dataset):
    def __init__(self, hs_patches, dm_patches, rd_patches, transform=None):
        self.hillshade_patches = hs_patches
        self.dem_patches = dm_patches
        self.roads_patches = rd_patches
        self.transform = transform
        # Compute binary labels: 1 if mean roads mask > threshold, else 0.
        self.labels = [1 if np.mean(mask[0] > 0) > 0.01 else 0 for mask in rd_patches]
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        print(f"Dataset class distribution: {pos} positive, {neg} negative patches.")

    def __len__(self):
        return len(self.hillshade_patches)

    def __getitem__(self, idx):
        hillshade = self.hillshade_patches[idx]
        dem = self.dem_patches[idx]
        roads = self.roads_patches[idx]

        hs_norm = normalize_channel(hillshade[0])
        dem_norm = normalize_channel(dem[0])
        hs_norm = hs_norm[np.newaxis, ...]
        dem_norm = dem_norm[np.newaxis, ...]
        
        input_img = np.concatenate([hs_norm, dem_norm, hs_norm], axis=0)
        input_img = np.nan_to_num(input_img, nan=0)
        target = np.nan_to_num(roads[0], nan=0)
        
        if self.transform:
            augmented = self.transform(image=input_img, mask=target)
            input_img = augmented['image']
            target = augmented['mask']
        
        input_tensor = torch.tensor(input_img, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.long)
        return input_tensor, target_tensor

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Splits the dataset into train, validation, and test sets.
    """
    total = len(dataset)
    train_size = int(train_ratio * total)
    val_size = int(val_ratio * total)
    test_size = total - train_size - val_size
    print(f"Dataset split -> Train: {train_size}, Val: {val_size}, Test: {test_size}")
    return random_split(dataset, [train_size, val_size, test_size],
                        generator=torch.Generator().manual_seed(seed))

def oversample_until_balanced(hs_patches, dem_patches, rd_patches, minority_threshold=0.01):
    """
    Balances the classes by oversampling positive patches until counts are equal.
    """
    pos_indices = [i for i, mask in enumerate(rd_patches) if np.mean(mask[0] > 0) > minority_threshold]
    num_pos = len(pos_indices)
    num_total = len(rd_patches)
    num_neg = num_total - num_pos
    print(f"Before oversampling: {num_pos} positive, {num_neg} negative patches.")
    if num_pos == 0:
        print("No positive samples found; cannot oversample.")
        return hs_patches, dem_patches, rd_patches
    required_extra = num_neg - num_pos
    copies_per_sample = int(np.floor(required_extra / num_pos))
    remainder = required_extra % num_pos
    print(f"Each positive patch will be duplicated {copies_per_sample} times, with {remainder} additional copies needed.")
    new_hs = list(hs_patches)
    new_dem = list(dem_patches)
    new_rd = list(rd_patches)
    for i in pos_indices:
        for _ in range(copies_per_sample):
            new_hs.append(hs_patches[i])
            new_dem.append(dem_patches[i])
            new_rd.append(rd_patches[i])
    for i in pos_indices[:remainder]:
        new_hs.append(hs_patches[i])
        new_dem.append(dem_patches[i])
        new_rd.append(rd_patches[i])
    new_pos = sum([1 for mask in new_rd if np.mean(mask[0] > 0) > minority_threshold])
    new_neg = len(new_rd) - new_pos
    print(f"After oversampling: {new_pos} positive, {new_neg} negative patches.")
    print(f"Total patches after oversampling: {len(new_hs)}")
    return new_hs, new_dem, new_rd

def create_augmented_dataset(hs_patches, dem_patches, rd_patches, augmentation_transform):
    """
    Creates an augmented dataset by concatenating the original and augmented datasets.
    """
    original_dataset = CreekDataset(hs_patches, dem_patches, rd_patches, transform=None)
    augmented_dataset = CreekDataset(hs_patches, dem_patches, rd_patches, transform=augmentation_transform)
    total = len(original_dataset) + len(augmented_dataset)
    print(f"Data augmentation: total dataset size after original+augmentation: {total}")
    return ConcatDataset([original_dataset, augmented_dataset])
