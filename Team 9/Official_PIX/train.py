import os
import argparse
import time
import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid

from model import Generator, PatchDiscriminator, GANLoss, numpy_to_tensor

# Parse arguments
parser = argparse.ArgumentParser(description='Pix2Pix Training Script for NumPy Dataset')
parser.add_argument('--data_root', type=str, default='./numpy_data', help='dataset root directory')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss (optional)')
parser.add_argument('--gan_mode', type=str, default='vanilla', help='vanilla | lsgan')
parser.add_argument('--save_interval', type=int, default=5, help='save model every N epochs')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='directory to save checkpoints')
parser.add_argument('--sample_dir', type=str, default='./samples', help='directory to save samples')
parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
parser.add_argument('--num_workers', type=int, default=4, help='number of worker threads for data loading')
args = parser.parse_args()

# Create directories
os.makedirs(args.checkpoint_dir, exist_ok=True)
os.makedirs(args.sample_dir, exist_ok=True)
os.makedirs(os.path.join(args.sample_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(args.sample_dir, 'val'), exist_ok=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize TensorBoard
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join('./logs', timestamp)
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs at {log_dir}")

# Custom Dataset class for NumPy files
class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', direction='AtoB'):
        self.root_dir = os.path.join(root_dir, mode)
        self.direction = direction
        self.input_files = sorted([f for f in os.listdir(self.root_dir) if f.startswith('input')])
        self.target_files = sorted([f for f in os.listdir(self.root_dir) if f.startswith('target')])
        
    def __len__(self):
        return len(self.input_files)
        
    def __getitem__(self, idx):
        input_path = os.path.join(self.root_dir, self.input_files[idx])
        target_path = os.path.join(self.root_dir, self.target_files[idx])
        input_data = np.load(input_path)
        target_data = np.load(target_path)
        
        input_tensor = numpy_to_tensor(input_data)
        target_tensor = numpy_to_tensor(target_data)
        
        if self.direction == 'AtoB':
            return {'input': input_tensor, 'target': target_tensor}
        else:
            return {'input': target_tensor, 'target': input_tensor}

def get_numpy_dataloader(root_dir, batch_size, mode='train', direction='AtoB', num_workers=4):
    dataset = NumpyDataset(root_dir, mode, direction)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), 
                                       num_workers=num_workers, drop_last=(mode == 'train'))

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def train():
    generator = Generator(in_channels=1, out_channels=1).to(device)
    discriminator = PatchDiscriminator(in_channels=2).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    criterion_GAN = GANLoss(gan_mode=args.gan_mode).to(device)
    criterion_BCE = nn.BCELoss()  # BCE for binary output
    criterion_L1 = nn.L1Loss()   # Optional L1 loss
    
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
    
    train_dataloader = get_numpy_dataloader(args.data_root, args.batch_size, 'train', args.direction, args.num_workers)
    val_dataloader = get_numpy_dataloader(args.data_root, 1, 'val', args.direction, args.num_workers)
    
    print(f"Training set: {len(train_dataloader.dataset)} samples")
    print(f"Validation set: {len(val_dataloader.dataset)} samples")
    
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_bce_loss = 0.0
        epoch_gan_loss = 0.0
        epoch_l1_loss = 0.0
        
        start_time = time.time()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                           desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, batch in progress_bar:
            real_A = batch['input'].to(device)  # Shape: [B, 1, 256, 256]
            real_B = batch['target'].to(device)  # Shape: [B, 1, 256, 256]
            
            # Forward pass
            fake_B = generator(real_A)  # Output in [0, 1] due to sigmoid
            
            # Train Discriminator
            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad()
            
            pred_real = discriminator(real_A, real_B)
            loss_D_real = criterion_GAN(pred_real, True)
            
            pred_fake = discriminator(real_A, fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            set_requires_grad(discriminator, False)
            optimizer_G.zero_grad()
            
            pred_fake = discriminator(real_A, fake_B)
            
            criterion_BCE = nn.BCELoss(reduction='none').to(device)

            # Inside the loop, update loss_G_BCE calculation
            loss_G_GAN = criterion_GAN(pred_fake, True)
            bce_loss = criterion_BCE(fake_B, real_B)
            weights = torch.ones_like(real_B)
            weights[real_B == 1] = 10.0  # Higher weight for boundary pixels
            loss_G_BCE = (bce_loss * weights).mean() * 10.0  # Apply weighting and scale
            loss_G_L1 = criterion_L1(fake_B, real_B) * 50.0
            loss_G = loss_G_GAN + loss_G_BCE + loss_G_L1
            
            loss_G.backward()
            optimizer_G.step()
            
            # Update metrics
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            epoch_bce_loss += loss_G_BCE.item()
            epoch_gan_loss += loss_G_GAN.item()
            epoch_l1_loss += loss_G_L1.item()
            
            progress_bar.set_postfix({
                'D_loss': f"{loss_D.item():.4f}",
                'G_loss': f"{loss_G.item():.4f}",
                'BCE': f"{loss_G_BCE.item():.4f}",
                'GAN': f"{loss_G_GAN.item():.4f}",
                'L1': f"{loss_G_L1.item():.4f}"
            })
            
            if i == 0:
                sample_images = torch.cat([real_A, fake_B, real_B], 3)  # No denormalization needed (already [0, 1])
                save_path = os.path.join(args.sample_dir, 'train', f'epoch_{epoch+1}.png')
                save_image(sample_images, save_path, nrow=1, normalize=False)
                grid = make_grid(sample_images, nrow=1, normalize=False)
                writer.add_image('Train/Samples', grid, epoch)
        
        # Average losses
        avg_d_loss = epoch_d_loss / len(train_dataloader)
        avg_g_loss = epoch_g_loss / len(train_dataloader)
        avg_bce_loss = epoch_bce_loss / len(train_dataloader)
        avg_gan_loss = epoch_gan_loss / len(train_dataloader)
        avg_l1_loss = epoch_l1_loss / len(train_dataloader)
        
        epoch_time = time.time() - start_time
        
        writer.add_scalar('Loss/train/Discriminator', avg_d_loss, epoch)
        writer.add_scalar('Loss/train/Generator', avg_g_loss, epoch)
        writer.add_scalar('Loss/train/BCE', avg_bce_loss, epoch)
        writer.add_scalar('Loss/train/GAN', avg_gan_loss, epoch)
        writer.add_scalar('Loss/train/L1', avg_l1_loss, epoch)
        writer.add_scalar('Time/epoch', epoch_time, epoch)
        
        print(f"Epoch [{epoch+1}/{args.epochs}] - "
              f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
              f"BCE: {avg_bce_loss:.4f}, GAN: {avg_gan_loss:.4f}, L1: {avg_l1_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        val_bce_loss = validate(generator, val_dataloader, epoch)
        
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'val_loss': val_bce_loss,
                'args': args
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest.pth'))
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'epoch_{epoch+1}.pth'))
            print(f"Checkpoint saved at epoch {epoch+1}")

def validate(generator, val_dataloader, epoch):
    generator.eval()
    val_bce_loss = 0.0
    
    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            real_A = batch['input'].to(device)
            real_B = batch['target'].to(device)
            
            fake_B = generator(real_A)
            loss = nn.BCELoss()(fake_B, real_B)  # BCE for validation
            val_bce_loss += loss.item()
            
            if i < 5:
                # Apply threshold for visualization (inference step)
                fake_B_binary = (fake_B > 0.5).float()
                img_sample = torch.cat([real_A, fake_B_binary, real_B], 3)
                save_path = os.path.join(args.sample_dir, 'val', f'epoch_{epoch+1}_sample_{i+1}.png')
                save_image(img_sample, save_path, normalize=False)
                if i == 0:
                    grid = make_grid(img_sample, nrow=1, normalize=False)
                    writer.add_image('Validation/Sample', grid, epoch)
    
    avg_val_loss = val_bce_loss / len(val_dataloader)
    writer.add_scalar('Loss/validation/BCE', avg_val_loss, epoch)
    print(f"Validation BCE Loss: {avg_val_loss:.4f}")
    
    generator.train()
    return avg_val_loss

if __name__ == "__main__":
    train()