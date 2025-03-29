import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNetDown(nn.Module):
    """U-Net downsampling block with optional dropout"""
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.down = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.down(x)

class UNetUp(nn.Module):
    """U-Net upsampling block with skip connections"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.up = nn.Sequential(*layers)
        
    def forward(self, x, skip_input):
        x = self.up(x)
        return torch.cat((x, skip_input), 1)

class Generator(nn.Module):
    """U-Net Generator with skip connections"""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator for pix2pix"""
    def __init__(self, in_channels=2):
        super().__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )
        
    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class GANLoss(nn.Module):
    """Define GAN objectives"""
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')
            
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
        
    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

def numpy_to_tensor(np_array):
    """
    Convert numpy array to PyTorch tensor with proper formatting for pix2pix
    
    Args:
        np_array: numpy array of shape (H,W,C)
        
    Returns:
        PyTorch tensor of shape (C,H,W) normalized to [0,1]
    """
    if np_array.dtype != np.float32:
        np_array = np_array.astype(np.float32)
        
    # Ensure shape is (H, W, C) and transpose to (C, H, W)
    if len(np_array.shape) == 3 and np_array.shape[2] == 1:
        np_array = np_array.transpose(2, 0, 1)  # Convert to (C,H,W)
    elif len(np_array.shape) != 3:
        raise ValueError(f"Expected 3D array (H,W,C), got shape {np_array.shape}")
    
    # Convert to PyTorch tensor without adding batch dimension
    tensor = torch.from_numpy(np_array)
    
    # Normalize to [0, 1] if in range [0, 255]
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    
    return tensor

if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 256)
    generator = Generator()
    gen_out = generator(x)
    print(f"Generator input: {x.shape}")
    print(f"Generator output: {gen_out.shape}")
    
    img_A = torch.randn(1, 1, 256, 256)
    img_B = torch.randn(1, 1, 256, 256)
    discriminator = PatchDiscriminator()
    disc_out = discriminator(img_A, img_B)
    print(f"Discriminator input A: {img_A.shape}, B: {img_B.shape}")
    print(f"Discriminator output: {disc_out.shape}")