import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import time
import os
import matplotlib.pyplot as plt
import numpy as np

# Set device: GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def bce_dice_loss(outputs, targets, smooth=1e-5):
    """
    Computes a combined loss: BCE + Dice Loss.
    Expects outputs of shape (B, 2, H, W) and targets of shape (B, H, W).
    """
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
    bce = nn.functional.binary_cross_entropy_with_logits(outputs, targets_onehot)
    
    outputs_prob = torch.sigmoid(outputs)
    intersection = (outputs_prob * targets_onehot).sum(dim=(2, 3))
    dice = 1 - (2 * intersection + smooth) / (outputs_prob.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3)) + smooth)
    dice_loss = dice.mean()
    return bce + dice_loss

def get_model(lr=1e-3, weight_decay=0.0):
    """
    Defines the Unet++ model, combined BCE+Dice loss, and optimizer.
    """
    model = smp.UnetPlusPlus(
        encoder_name='densenet121',
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, bce_dice_loss, optimizer

def training_loop(model, dataloader, loss_fn, optimizer, num_epochs):
    """
    Training loop that prints statistics.
    """
    model.train()
    print("\n========== START TRAINING ==========")
    for epoch in range(num_epochs):
        print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
        for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            print(f"\nBatch {batch_idx}: inputs.shape: {inputs.shape}, targets.shape: {targets.shape}")
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                print("  [WARNING] NaNs detected! Skipping batch...")
                continue
            outputs = model(inputs)
            if torch.isnan(outputs).any():
                print("  [WARNING] NaNs in model outputs. Skipping backward pass.")
                continue
            loss = loss_fn(outputs, targets)
            if torch.isnan(loss):
                print("  [WARNING] NaN loss! Skipping backward pass.")
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  Loss: {loss.item():.4f}")
    print("\n========== TRAINING COMPLETE ==========")

def evaluation_loop(model, dataloader, loss_fn):
    """
    Evaluation loop that computes average loss.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)
    avg_loss = total_loss / count if count > 0 else float('inf')
    print(f"\nEvaluation Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss

def train_with_early_stopping(model, train_loader, val_loader, loss_fn, optimizer,
                              num_epochs, patience=3, model_save_path="best_model.pth"):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5, verbose=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
        
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  Batch {batch_idx} Loss: {loss.item():.4f}")
        
        val_loss = evaluation_loop(model, val_loader, loss_fn)
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        estimated_total = total_elapsed / (epoch + 1) * num_epochs
        eta = estimated_total - total_elapsed
        print(f"Epoch {epoch+1} time: {epoch_time:.2f}s, Total elapsed: {total_elapsed:.2f}s, ETA: {eta:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")

def save_predictions(model, dataset, save_dir, indices):
    """
    Saves prediction figures for the given indices.
    Each figure shows the input image, predicted mask, and ground truth.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    for idx in indices:
        input_tensor, target_tensor = dataset[idx]
        input_batch = input_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_batch)
        pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()
        input_img = input_tensor.cpu().numpy().transpose(1, 2, 0)
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(input_img)
        axs[0].set_title(f"Input Image (idx {idx})")
        axs[1].imshow(pred_mask, cmap='gray')
        axs[1].set_title("Predicted Mask")
        axs[2].imshow(target_tensor.cpu().numpy(), cmap='gray')
        axs[2].set_title("Ground Truth")
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"prediction_{idx}.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved prediction for index {idx} at {save_path}")
    model.train()
