import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class NumpyDataset(Dataset):
    """Dataset class for loading NumPy arrays for Pix2Pix."""
    
    def __init__(self, root_dir, mode='train', direction='AtoB', img_size=256, transform=None):
        """
        Args:
            root_dir (str): Root directory containing 'train', 'val', 'test' subdirectories
            mode (str): 'train', 'val', or 'test'
            direction (str): 'AtoB' (input -> target) or 'BtoA' (target -> input)
            img_size (int): Size of the images (for compatibility with train.py)
            transform (callable, optional): Optional transform to be applied on the sample
        """
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.direction = direction
        self.img_size = img_size
        self.is_train = (mode == 'train')
        
        # Find all input files and their corresponding target files
        self.input_files = sorted([
            os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir)
            if f.startswith('input_') and f.endswith('.npy')
        ])
        
        self.target_files = []
        for input_file in self.input_files:
            idx = os.path.basename(input_file).split('_')[1].split('.')[0]
            target_file = os.path.join(self.root_dir, f'target_{idx}.npy')
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"Target file not found for {input_file}")
            self.target_files.append(target_file)
        
        print(f"Found {len(self.input_files)} {mode} samples")
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        # Load input and target numpy arrays
        input_array = np.load(self.input_files[idx])  # Shape: [256, 256, 1]
        target_array = np.load(self.target_files[idx])  # Shape: [256, 256, 1]
        
        # Ensure single channel (remove any RGB conversion)
        if input_array.shape[-1] != 1:
            input_array = input_array[..., :1]  # Take first channel if more exist
        if target_array.shape[-1] != 1:
            target_array = target_array[..., :1]  # Take first channel if more exist
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float()
        
        # Permute dimensions from HWC to CHW format as PyTorch expects
        input_tensor = input_tensor.permute(2, 0, 1)   # From [256, 256, 1] to [1, 256, 256]
        target_tensor = target_tensor.permute(2, 0, 1) # From [256, 256, 1] to [1, 256, 256]
        
        # Apply transformations if training (e.g., random flip)
        if self.transform and self.is_train:
            if random.random() > 0.5:
                input_tensor = torch.flip(input_tensor, [2])  # Flip horizontally
                target_tensor = torch.flip(target_tensor, [2])  # Flip horizontally
        
        # Data is already in [0, 1] from preprocessing, no further normalization needed
        return {
            'input': input_tensor,
            'target': target_tensor,
            'input_path': self.input_files[idx],
            'target_path': self.target_files[idx]
        }

def get_numpy_dataloader(root_dir, batch_size=1, mode='train', direction='AtoB', num_workers=4, img_size=256):
    """
    Creates a DataLoader for the NumPy dataset.
    
    Args:
        root_dir (str): Directory containing 'train', 'val', 'test' subdirectories
        batch_size (int): Batch size
        mode (str): 'train', 'val', or 'test'
        direction (str): 'AtoB' (input -> target) or 'BtoA' (target -> input)
        num_workers (int): Number of worker threads for loading data
        img_size (int): Size of the images (not used, but kept for compatibility)
    """
    dataset = NumpyDataset(
        root_dir=root_dir,
        mode=mode,
        direction=direction,
        img_size=img_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        drop_last=(mode == 'train'),
        pin_memory=True
    )
    
    return dataloader

# Alias for compatibility with train.py
get_dataloader = get_numpy_dataloader

if __name__ == "__main__":
    dataloader = get_numpy_dataloader(
        root_dir='./numpy_data',
        batch_size=4,
        mode='train',
        direction='AtoB'
    )
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    for batch in dataloader:
        inputs = batch['input']
        targets = batch['target']
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        break