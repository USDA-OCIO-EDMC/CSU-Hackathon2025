import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid, save_image

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all parameters in a network"""
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def tensor_to_image(tensor, threshold=False):
    """Convert single-channel tensor to PIL Image
    
    Args:
        tensor (torch.Tensor): Image tensor with shape [C, H, W] (C=1 for grayscale)
        threshold (bool): Apply threshold at 0.5 for binary output visualization
        
    Returns:
        PIL.Image: Converted PIL image
    """
    tensor = tensor.clone().detach().cpu()
    if tensor.dim() == 4:  # Remove batch dimension if present
        tensor = tensor.squeeze(0)
    
    if threshold:  # For binary output visualization
        tensor = (tensor > 0.5).float()
    
    tensor = torch.clamp(tensor, 0, 1)  # Ensure [0, 1] range
    np_array = tensor.numpy().squeeze(0)  # From [1, H, W] to [H, W]
    np_array = (np_array * 255).astype(np.uint8)
    image = Image.fromarray(np_array, mode='L')  # Grayscale mode
    return image

def visualize_results(real_A, real_B, fake_B, figsize=(15, 5), save_path=None):
    """Visualize training results: input, ground truth, generated
    
    Args:
        real_A (torch.Tensor): Input image tensor [B, 1, H, W]
        real_B (torch.Tensor): Ground truth image tensor [B, 1, H, W]
        fake_B (torch.Tensor): Generated image tensor [B, 1, H, W]
        figsize (tuple): Figure size for the plot
        save_path (str, optional): Path to save the visualization
    """
    if real_A.dim() == 4 and real_A.size(0) > 1:
        real_A = real_A[0]
        real_B = real_B[0]
        fake_B = fake_B[0]
    
    real_A_img = tensor_to_image(real_A)  # No thresholding for input
    real_B_img = tensor_to_image(real_B, threshold=True)  # Binary ground truth
    fake_B_img = tensor_to_image(fake_B, threshold=True)  # Binary generated output
    
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].imshow(real_A_img, cmap='gray')
    ax[0].set_title('Input (A)')
    ax[0].axis('off')
    
    ax[1].imshow(fake_B_img, cmap='gray')
    ax[1].set_title('Generated (B)')
    ax[1].axis('off')
    
    ax[2].imshow(real_B_img, cmap='gray')
    ax[2].set_title('Ground Truth (B)')
    ax[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def save_checkpoint(model_G, model_D, optimizer_G, optimizer_D, epoch, save_path, args=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'generator': model_G.state_dict(),
        'discriminator': model_D.state_dict(),
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict()
    }
    if args is not None:
        checkpoint['args'] = args
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(checkpoint_path, model_G, model_D=None, optimizer_G=None, optimizer_D=None):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None, 0
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_G.load_state_dict(checkpoint['generator'])
    if model_D and 'discriminator' in checkpoint:
        model_D.load_state_dict(checkpoint['discriminator'])
    if optimizer_G and 'optimizer_G' in checkpoint:
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    if optimizer_D and 'optimizer_D' in checkpoint:
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    epoch = checkpoint.get('epoch', 0)
    args = checkpoint.get('args', None)
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return args, epoch

def save_image_grid(images, paths, nrow=1, padding=2, normalize=False, save_path=None):
    """Create and save a grid of images"""
    if normalize:  # If normalize=True, assume input is not yet in [0, 1]
        images = images.clone()
        images = (images + 1) / 2.0
    images = torch.clamp(images, 0, 1)
    # Apply thresholding for binary outputs (except for input)
    if 'target' in paths[0] or 'generated' in paths[0].lower():
        images = (images > 0.5).float()
    grid = make_grid(images, nrow=nrow, padding=padding, normalize=False)
    if save_path:
        save_image(grid, save_path)
    grid_np = grid.detach().cpu().numpy().transpose(1, 2, 0)
    return grid_np

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images (higher is better)"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr