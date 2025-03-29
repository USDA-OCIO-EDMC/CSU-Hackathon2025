import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from model import Generator
from dataset import get_numpy_dataloader
from utils import calculate_psnr

# Parse command line arguments
parser = argparse.ArgumentParser(description='Pix2Pix Evaluation Script')
parser.add_argument('--data_root', type=str, default='./numpy_data', help='dataset root directory')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/latest.pth', help='path to model checkpoint (default: best.pth)')
parser.add_argument('--output_dir', type=str, default='./results', help='directory to save results')
parser.add_argument('--num_workers', type=int, default=4, help='number of worker threads')
parser.add_argument('--save_numpy', action='store_true', help='save results as numpy arrays instead of images')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def evaluate():
    # Initialize model with 1-channel input and output
    generator = Generator(in_channels=1, out_channels=1).to(device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        try:
            checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
            generator.load_state_dict(checkpoint['generator'])
            epoch = checkpoint.get('epoch', 'unknown')
            best_val_bce_loss = checkpoint.get('best_val_bce_loss', 'N/A')
            print(f"Loaded checkpoint from {args.checkpoint_path} (epoch {epoch}, best_val_bce_loss: {best_val_bce_loss})")
        except Exception as e:
            print(f"Error loading checkpoint with default settings: {e}")
            print("Trying with alternative loading method...")
            try:
                state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
                if isinstance(state_dict, dict) and 'generator' in state_dict:
                    generator.load_state_dict(state_dict['generator'])
                    epoch = state_dict.get('epoch', 'unknown')
                    print(f"Successfully loaded generator weights (epoch {epoch})")
                else:
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
    
    # Create test dataloader
    test_dataloader = get_numpy_dataloader(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        mode='test',
        direction=args.direction,
        num_workers=args.num_workers
    )
    
    print(f"Test set: {len(test_dataloader.dataset)} samples")
    
    # Evaluation metrics
    total_psnr = 0.0
    total_bce = 0.0
    criterion_BCE = nn.BCELoss().to(device)
    
    # Create subdirectories for different visualizations
    input_dir = os.path.join(args.output_dir, 'input')
    generated_dir = os.path.join(args.output_dir, 'generated')
    target_dir = os.path.join(args.output_dir, 'target')
    comparison_dir = os.path.join(args.output_dir, 'comparison')
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run evaluation
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            # Get data
            real_A = batch['input'].to(device)  # [B, 1, 256, 256], [0, 1]
            real_B = batch['target'].to(device)  # [B, 1, 256, 256], [0, 1]
            file_paths = batch['input_path']
            
            # Forward pass
            fake_B = generator(real_A)  # [B, 1, 256, 256], [0, 1]
            
            # Calculate metrics
            fake_B_binary = (fake_B > 0.5).float()  # Threshold for binary output
            bce_loss = criterion_BCE(fake_B, real_B)  # BCE on continuous output
            psnr = calculate_psnr(real_B.cpu().numpy(), fake_B_binary.cpu().numpy())  # PSNR on binary output
            
            total_bce += bce_loss.item()
            total_psnr += psnr
            
            # Get base filename
            base_name = os.path.basename(file_paths[0]).split('.')[0]
            
            # Visualization (already in [0, 1])
            real_A_vis = real_A
            real_B_vis = real_B
            fake_B_vis = fake_B_binary  # Use binary for visualization
            
            # Save results
            if args.save_numpy:
                np.save(os.path.join(input_dir, f"{base_name}_input.npy"), 
                        real_A_vis.cpu().numpy())
                np.save(os.path.join(target_dir, f"{base_name}_target.npy"), 
                        real_B_vis.cpu().numpy())
                np.save(os.path.join(generated_dir, f"{base_name}_generated.npy"), 
                        fake_B_vis.cpu().numpy())
            else:
                save_image(real_A_vis, os.path.join(input_dir, f"{base_name}_input.png"))
                save_image(fake_B_vis, os.path.join(generated_dir, f"{base_name}_generated.png"))
                save_image(real_B_vis, os.path.join(target_dir, f"{base_name}_target.png"))
                
                # Create comparison image
                comparison = torch.cat([real_A_vis, fake_B_vis, real_B_vis], dim=3)
                save_image(comparison, os.path.join(comparison_dir, f"{base_name}_comparison.png"))
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataloader)} samples - PSNR: {psnr:.2f} dB, BCE: {bce_loss.item():.4f}")
    
    # Calculate average metrics
    avg_psnr = total_psnr / len(test_dataloader)
    avg_bce = total_bce / len(test_dataloader)
    print(f"Evaluation complete. Average PSNR: {avg_psnr:.2f} dB, Average BCE: {avg_bce:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average BCE: {avg_bce:.4f}\n")
        f.write(f"Total samples: {len(test_dataloader.dataset)}\n")
        f.write(f"Model checkpoint: {args.checkpoint_path}\n")

def main():
    evaluate()

if __name__ == "__main__":
    main()