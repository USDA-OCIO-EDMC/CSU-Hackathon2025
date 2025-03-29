#!/usr/bin/env python3
# single_channel_road_data.py - Creates 400 augmented patches using only the first channel of input data
# Modified version to use only the first band of the stacked raster

import numpy as np
import os
import shutil
import random
from tqdm import tqdm
import gc  # Garbage collector
import albumentations as albu

# Delete directories if they exist and recreate them
def recreate_dir(dir_path):
    if os.path.exists(dir_path):
        print(f"Removing existing directory: {dir_path}")
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"Created directory: {dir_path}")

# Create output directories
recreate_dir('numpy_data')
recreate_dir('numpy_data/train')
recreate_dir('numpy_data/val')
recreate_dir('numpy_data/test')

# Load the data in chunks or with memory mapping
print("Loading data with memory mapping...")
try:
    # Use memory mapping for large files
    input_raster = np.load('stacked_rasters.npy', mmap_mode='r')
    road_mask = np.load('road_mask.npy', mmap_mode='r')
except:
    print("Memory mapping failed, falling back to standard loading...")
    input_raster = np.load('stacked_rasters.npy')
    road_mask = np.load('road_mask.npy')

# Examine shapes
print(f"Input shape: {input_raster.shape}")
print(f"Mask shape: {road_mask.shape}")

# Fix axes order if needed
needs_transpose = False
if len(input_raster.shape) == 3 and input_raster.shape[2] != 6:
    needs_transpose = True
    print("Will need to transpose data when extracting patches")

# Calculate normalization params for just the first channel
print("Calculating normalization parameters for first channel only...")
# Sample a subset of points to estimate min/max
sample_points = 10000
h, w = road_mask.shape[:2]
indices = [(random.randint(0, h-1), random.randint(0, w-1)) for _ in range(sample_points)]

if needs_transpose:
    # For transposed data, get only first band
    samples = np.array([input_raster[0, y, x] for y, x in indices])
    # We only need min/max for the first band
    input_min = samples.min()
    input_max = samples.max()
else:
    # For correctly oriented data, get only first band
    samples = np.array([input_raster[y, x, 0] for y, x in indices])
    # We only need min/max for the first band
    input_min = samples.min()
    input_max = samples.max()

print(f"First channel normalization parameters: min={input_min}, max={input_max}")

del samples
gc.collect()  # Force garbage collection

# Set parameters for data splitting
target_total_patches = 400
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate target sizes for each set
target_train = int(target_total_patches * train_ratio)
target_val = int(target_total_patches * val_ratio)
target_test = target_total_patches - target_train - target_val

print(f"Target distribution: {target_train} train, {target_val} validation, {target_test} test patches")

# Number of original patches to extract (we'll augment these later)
num_base_patches = 100  # We'll augment these to get 400 total
print(f"Will extract {num_base_patches} base patches and augment to {target_total_patches} total...")

# Define augmentation transforms
def get_transforms():
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.RandomBrightnessContrast(p=0.2),
        albu.GaussNoise(p=0.2)
    ])

# Find non-empty patches (using 256x256 patches)
def find_non_empty_patches(mask, patch_size=256, min_road_pixels=20, max_patches=2000):
    h, w = mask.shape
    valid_patches = []
    step = patch_size // 2  # 50% overlap
    
    # Calculate number of steps in each dimension
    y_steps = (h - patch_size) // step + 1
    x_steps = (w - patch_size) // step + 1
    
    # Generate random indices to check instead of searching exhaustively
    indices = [(y*step, x*step) 
               for y in range(min(y_steps, 50))  # Limit search space
               for x in range(min(x_steps, 50))]
    
    random.shuffle(indices)  # Randomize order
    
    count = 0
    pbar = tqdm(total=min(len(indices), max_patches), desc="Finding patches")
    
    for y, x in indices:
        if y + patch_size > h or x + patch_size > w:
            continue
            
        # Check if patch contains enough road pixels
        patch = mask[y:y+patch_size, x:x+patch_size]
        if np.sum(patch) >= min_road_pixels:
            valid_patches.append((y, x))
            
        count += 1
        pbar.update(1)
        
        if count >= max_patches or len(valid_patches) >= num_base_patches:
            break
            
    pbar.close()
    return valid_patches

# Find patches containing roads (with a limit on how many patches to check)
print("Finding patches with roads...")
valid_patches = find_non_empty_patches(road_mask, max_patches=1000)
print(f"Found {len(valid_patches)} patches containing roads")

if len(valid_patches) == 0:
    print("No patches with sufficient road pixels found. Using random patches...")
    h, w = road_mask.shape
    patch_size = 256
    valid_patches = [(random.randint(0, h - patch_size), 
                     random.randint(0, w - patch_size)) 
                     for _ in range(num_base_patches)]

# Set random seed for reproducibility
random.seed(42)

# If we have more valid patches than needed, randomly select subset
if len(valid_patches) > num_base_patches:
    valid_patches = random.sample(valid_patches, num_base_patches)

# Calculate number of augmentations needed per patch
augmentations_per_patch = max(1, target_total_patches // len(valid_patches))
print(f"Will create {augmentations_per_patch} augmented versions of each base patch")

# Prepare transforms
transform = get_transforms()

# Pre-allocate patches to datasets before processing
# This ensures we get the right distribution
patch_indices = list(range(len(valid_patches) * augmentations_per_patch))
random.shuffle(patch_indices)

# Create lists to track which set each patch will go to
train_indices = patch_indices[:target_train]
val_indices = patch_indices[target_train:target_train + target_val]
test_indices = patch_indices[target_train + target_val:]

print(f"Allocated {len(train_indices)} indices for train")
print(f"Allocated {len(val_indices)} indices for validation")
print(f"Allocated {len(test_indices)} indices for test")

# Create dictionaries to store which set each patch belongs to
patch_assignments = {}
for idx in train_indices:
    patch_assignments[idx] = 'train'
for idx in val_indices:
    patch_assignments[idx] = 'val'
for idx in test_indices:
    patch_assignments[idx] = 'test'

# Process and save patches
print("Extracting, augmenting and saving patches...")
train_count = val_count = test_count = 0
counters = {'train': 0, 'val': 0, 'test': 0}

pbar = tqdm(total=min(len(patch_assignments), target_total_patches))

patch_count = 0
for base_idx, (y, x) in enumerate(valid_patches):
    patch_size = 256
    
    try:
        # Extract ONLY the first channel patch (differently handling transposed data if needed)
        if needs_transpose:
            # For transposed data, only get the first channel
            img_patch = input_raster[0, y:y+patch_size, x:x+patch_size].copy()
            # Reshape to [H, W, 1]
            img_patch = img_patch.reshape(patch_size, patch_size, 1)
        else:
            # For correctly oriented data, only get the first channel
            img_patch = input_raster[y:y+patch_size, x:x+patch_size, 0].copy()
            # Reshape to [H, W, 1]
            img_patch = img_patch.reshape(patch_size, patch_size, 1)
        
        mask_patch = road_mask[y:y+patch_size, x:x+patch_size].copy()
        
        # Skip if patch is incomplete
        if img_patch.shape[0] < patch_size or img_patch.shape[1] < patch_size:
            continue
        
        # Normalize patch (just one channel now)
        img_patch = (img_patch - input_min) / max(input_max - input_min, 1e-5)
        img_patch = np.clip(img_patch, 0, 1)  # Ensure values are in [0,1]
        
        # Convert mask to binary (0 or 1) and reshape to (256, 256, 1)
        mask_patch = (mask_patch > 0.5).astype(np.float32)
        mask_patch = mask_patch.reshape(256, 256, 1)
        
        # Create base patch + augmented versions
        for aug_idx in range(augmentations_per_patch):
            current_idx = base_idx * augmentations_per_patch + aug_idx
            
            # Skip if we've reached the target
            if current_idx >= len(patch_assignments):
                continue
                
            # Get the pre-determined set for this patch
            save_dir = patch_assignments[current_idx]
            idx = counters[save_dir]
            counters[save_dir] += 1
            
            if aug_idx == 0:
                # Use original patch without augmentation
                aug_img = img_patch.copy()
                aug_mask = mask_patch.copy()
            else:
                # Apply augmentation - must use compatible transforms for single-channel
                augmented = transform(image=img_patch, mask=mask_patch)
                aug_img = augmented['image']
                aug_mask = augmented['mask']
            
            # Save directly to disk - avoid storing in memory
            np.save(f'numpy_data/{save_dir}/input_{idx:03d}.npy', aug_img)
            np.save(f'numpy_data/{save_dir}/target_{idx:03d}.npy', aug_mask)
            
            patch_count += 1
            pbar.update(1)
            
            # Debug output for test set
            if save_dir == 'test':
                print(f"Saved test patch {idx} at location ({y}, {x}) with aug_idx {aug_idx}")
            
            # Break if we've reached the target number of patches
            if patch_count >= target_total_patches:
                break
        
    except Exception as e:
        print(f"Error processing patch at ({y}, {x}): {e}")
    
    # Force garbage collection occasionally
    if base_idx % 10 == 0:
        gc.collect()
    
    # Break if we've reached the target number of patches
    if patch_count >= target_total_patches:
        break

pbar.close()

print(f"Extracted and saved {patch_count} patches")
print(f"Train samples: {counters['train']}")
print(f"Validation samples: {counters['val']}")
print(f"Test samples: {counters['test']}")

# Verify the saved datasets
train_inputs = len([f for f in os.listdir('numpy_data/train') if f.startswith('input')])
val_inputs = len([f for f in os.listdir('numpy_data/val') if f.startswith('input')])
test_inputs = len([f for f in os.listdir('numpy_data/test') if f.startswith('input')])

print(f"Train samples verified: {train_inputs}")
print(f"Validation samples verified: {val_inputs}")
print(f"Test samples verified: {test_inputs}")
print(f"Total: {train_inputs + val_inputs + test_inputs}")

# Check if any set is missing
if test_inputs == 0:
    print("\nWARNING: No test files were created!")
    print("Creating 10 sample test files as a fallback...")
    
    # Create sample test files as a last resort
    for i in range(10):
        # Create an empty sample - now single channel
        sample_img = np.zeros((256, 256, 1), dtype=np.float32)
        sample_mask = np.zeros((256, 256, 1), dtype=np.float32)
        
        try:
            np.save(f'numpy_data/test/input_{i:03d}.npy', sample_img)
            np.save(f'numpy_data/test/target_{i:03d}.npy', sample_mask)
            print(f"Created emergency test sample {i}")
        except Exception as e:
            print(f"Error creating emergency test sample: {e}")

# Save dataset info
with open('dataset_info.txt', 'w') as f:
    f.write(f"Total patches: {patch_count}\n")
    f.write(f"Train samples: {counters['train']}\n")
    f.write(f"Validation samples: {counters['val']}\n")
    f.write(f"Test samples: {counters['test']}\n")
    f.write(f"Input shape: (256, 256, 1)\n")  # Now single channel
    f.write(f"Target shape: (256, 256, 1)\n")
    f.write("\nData format: NumPy arrays\n")
    f.write(f"- Each input is a separate .npy file with shape (256, 256, 1)\n")  # Updated
    f.write("- Each target is a separate .npy file with shape (256, 256, 1)\n")
    f.write("\nAugmentations applied:\n")
    f.write("- Horizontal flip (50% probability)\n")
    f.write("- Vertical flip (50% probability)\n")
    f.write("- 90-degree rotation (50% probability)\n") 
    f.write("- Random brightness/contrast (20% probability)\n")
    f.write("- Gaussian noise (20% probability)\n")

print("\nData preparation complete!")
print("All data saved with dimensions:")
print("- Inputs: (256, 256, 1)")  # Now single channel
print("- Targets: (256, 256, 1)")