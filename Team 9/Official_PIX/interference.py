import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from model import Generator

# Parse command line arguments
parser = argparse.ArgumentParser(description='Pix2Pix Inference Script')
parser.add_argument('--input_file', type=str, required=True, help='path to input numpy file')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/latest.pth', 
                    help='path to model checkpoint')
parser.add_argument('--output_dir', type=str, default='./inference_results', 
                    help='directory to save results')
parser.add_argument('--visualize', action='store_true', help='visualize results using matplotlib')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def inference():
    # Initialize model with 3-channel input/output for RGB
    generator = Generator(in_channels=3, out_channels=3).to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        try:
            # First try with weights_only=False (for compatibility with older PyTorch versions)
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            generator.load_state_dict(checkpoint['generator'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded checkpoint from {args.checkpoint_path} (epoch {epoch})")
        except Exception as e:
            print(f"Error loading checkpoint with default settings: {e}")
            print("Trying with alternative loading method...")
            
            try:
                # If the above fails, try direct state dictionary loading
                state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
                
                # If it's a nested dictionary with 'generator' key
                if isinstance(state_dict, dict) and 'generator' in state_dict:
                    generator.load_state_dict(state_dict['generator'])
                    epoch = state_dict.get('epoch', 'unknown')
                    print(f"Successfully loaded generator weights (epoch {epoch})")
                # If it's the raw state dictionary
                else:
                    # Try to load it directly
                    generator.load_state_dict(state_dict)
                    print("Successfully loaded generator weights (epoch unknown)")
            except Exception as e2:
                print(f"Error loading checkpoint with alternative method: {e2}")
                print("Please check your checkpoint file format")
                return
    else:
        print(f"No checkpoint found at {args.checkpoint_path}, exiting...")
        return
    
    # Set model to evaluation mode
    generator.eval()
    
    # Load input file
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} not found, exiting...")
        return
    
    # Load numpy file
    try:
        input_array = np.load(args.input_file)
        print(f"Loaded input array with shape: {input_array.shape}")
        
        # Make sure it has the right dimensions for RGB (3 channels)
        if len(input_array.shape) == 3 and input_array.shape[-1] == 1:
            # Convert single channel to 3 channels by repeating
            input_array = np.repeat(input_array, 3, axis=2)
            print(f"Converted single-channel input to RGB with shape: {input_array.shape}")
        elif len(input_array.shape) == 2:
            # Add channel dimension if it's just a 2D array
            input_array = np.stack([input_array] * 3, axis=-1)
            print(f"Added RGB channel dimension, new shape: {input_array.shape}")
        elif len(input_array.shape) == 3 and input_array.shape[-1] != 3:
            print(f"Expected 3 channels for RGB input, got {input_array.shape[-1]}")
            # Try to take the first 3 channels if there are more
            if input_array.shape[-1] > 3:
                input_array = input_array[:, :, :3]
                print(f"Using first 3 channels for RGB, new shape: {input_array.shape}")
            else:
                # Try to reshape if possible
                input_array = np.repeat(input_array[:, :, :1], 3, axis=2)
                print(f"Reshaped input array to {input_array.shape}")
        
        # Convert to PyTorch tensor
        input_tensor = torch.from_numpy(input_array).float()
        
        # Handle dimensions
        if len(input_tensor.shape) == 3:  # [H, W, C]
            input_tensor = input_tensor.permute(2, 0, 1)  # [C, H, W]
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        
        # Normalize to [-1, 1] as expected by the model
        if input_tensor.max() <= 1.0:
            input_tensor = input_tensor * 2.0 - 1.0
        elif input_tensor.max() > 1.0:
            input_tensor = input_tensor / 127.5 - 1.0
        
        print(f"Prepared input tensor with shape: {input_tensor.shape}")
        
    except Exception as e:
        print(f"Error loading or processing input file: {e}")
        return
    
    # Run inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        generated = generator(input_tensor)
        
        # Denormalize output from [-1, 1] to [0, 1]
        output_tensor = (generated + 1) / 2.0
        input_tensor_norm = (input_tensor + 1) / 2.0
        
        # Get base filename without extension
        base_name = os.path.basename(args.input_file).split('.')[0]
        
        # Save input and generated output
        input_path = os.path.join(args.output_dir, f"{base_name}_input.png")
        output_path = os.path.join(args.output_dir, f"{base_name}_output.png")
        save_image(input_tensor_norm, input_path, normalize=False)
        save_image(output_tensor, output_path, normalize=False)
        
        # Create and save a comparison image (input and output)
        comparison = torch.cat([input_tensor_norm, output_tensor], 3)
        comparison_path = os.path.join(args.output_dir, f"{base_name}_comparison.png")
        save_image(comparison, comparison_path, normalize=False)
        
        print(f"Results saved to {args.output_dir}")
        
        # Visualize results
        if args.visualize:
            plt.figure(figsize=(12, 6))
            
            # Plot input
            plt.subplot(1, 2, 1)
            input_img = input_tensor_norm[0].cpu().permute(1, 2, 0).numpy()
            plt.imshow(input_img)
            plt.title("Input Image")
            plt.axis('off')
            
            # Plot output
            plt.subplot(1, 2, 2)
            output_img = output_tensor[0].cpu().permute(1, 2, 0).numpy()
            plt.imshow(output_img)
            plt.title("Generated Output")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()

def main():
    inference()

if __name__ == "__main__":
    main()