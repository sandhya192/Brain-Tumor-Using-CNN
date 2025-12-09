"""
Brain Tumor Segmentation Training Script
BRISC 2025 Dataset - Pixel-wise Tumor Segmentation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime


class BrainTumorSegmentationDataset(Dataset):
    """Custom Dataset for Brain Tumor Segmentation"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.images_dir = os.path.join(root_dir, 'segmentation_task', split, 'images')
        self.masks_dir = os.path.join(root_dir, 'segmentation_task', split, 'masks')
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Loaded {len(self.image_files)} images from {split} set")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load mask (convert .jpg to .png extension if needed)
        mask_name = img_name.replace('.jpg', '.png')
        mask_path = os.path.join(self.masks_dir, mask_name)
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        if self.transform:
            # Apply same transform to both image and mask
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            image = self.transform['image'](image)
            
            torch.manual_seed(seed)
            mask = self.transform['mask'](mask)
            
        # Binarize mask (0 or 1)
        mask = (mask > 0.5).float()
        
        return image, mask


def get_segmentation_transforms(img_size=256, augment=True):
    """
    Get data preprocessing and augmentation transforms for segmentation
    
    Args:
        img_size (int): Target image size
        augment (bool): Whether to apply data augmentation
    
    Returns:
        dict: Dictionary with 'train' and 'test' transforms
    """
    if augment:
        train_img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor()
        ])
        
        train_transform = {'image': train_img_transform, 'mask': train_mask_transform}
    else:
        train_img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        train_mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        train_transform = {'image': train_img_transform, 'mask': train_mask_transform}
    
    test_img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    test_transform = {'image': test_img_transform, 'mask': test_mask_transform}
    
    return {'train': train_transform, 'test': test_transform}


class UNet(nn.Module):
    """U-Net architecture for image segmentation"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1]*2)
        
        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._block(feature*2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx//2]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = transforms.functional.resize(x, size=skip.shape[2:])
            
            concat_skip = torch.cat((skip, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE and Dice Loss"""
    
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        return self.alpha * self.bce(pred, target) + (1 - self.alpha) * self.dice(pred, target)


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate segmentation metrics"""
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target_binary.view(-1)
    
    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2. * intersection) / (union + 1e-8)
    iou = intersection / (union - intersection + 1e-8)
    
    return dice.item(), iou.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        dice, iou = calculate_metrics(outputs, masks)
        running_dice += dice
        running_iou += iou
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'dice': f'{running_dice/len(pbar):.4f}',
            'iou': f'{running_iou/len(pbar):.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Calculate metrics
            dice, iou = calculate_metrics(outputs, masks)
            running_dice += dice
            running_iou += iou
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'dice': f'{running_dice/len(pbar):.4f}',
                'iou': f'{running_iou/len(pbar):.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_dice, epoch_iou


def visualize_predictions(model, dataset, device, num_samples=5, save_path='predictions.png'):
    """Visualize model predictions"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            pred = model(image_tensor).squeeze().cpu().numpy()
            
            # Denormalize image for display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predictions saved to {save_path}")


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice Score
    axes[1].plot(history['train_dice'], label='Train Dice')
    axes[1].plot(history['val_dice'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    # IoU
    axes[2].plot(history['train_iou'], label='Train IoU')
    axes[2].plot(history['val_iou'], label='Val IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('Intersection over Union')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def main():
    # Configuration
    config = {
        'data_root': r'd:\Brain Tumor Project\brisc2025',
        'batch_size': 16,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'img_size': 256,
        'num_workers': 4,
        'augmentation': True,
        'patience': 15,
        'output_dir': 'outputs/segmentation'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get transforms
    transforms_dict = get_segmentation_transforms(
        img_size=config['img_size'],
        augment=config['augmentation']
    )
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = BrainTumorSegmentationDataset(
        config['data_root'],
        split='train',
        transform=transforms_dict['train']
    )
    
    test_dataset = BrainTumorSegmentationDataset(
        config['data_root'],
        split='test',
        transform=transforms_dict['test']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\n=== Creating Model ===")
    model = UNet(in_channels=3, out_channels=1)
    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # Training loop
    print("\n=== Starting Training ===")
    history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': []
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model, test_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'config': config
            }, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"✓ Best model saved! Val Dice: {val_dice:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    print("\n=== Final Evaluation ===")
    checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_dice, test_iou = validate(model, test_loader, criterion, device)
    
    print(f"\nBest Validation Dice: {best_val_dice:.4f}")
    print(f"Final Test Dice: {test_dice:.4f}")
    print(f"Final Test IoU: {test_iou:.4f}")
    
    # Visualize predictions
    visualize_predictions(
        model, test_dataset, device, num_samples=5,
        save_path=os.path.join(config['output_dir'], 'predictions.png')
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(config['output_dir'], 'training_history.png')
    )
    
    # Save history
    with open(os.path.join(config['output_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n=== Training Complete ===")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()
