"""
Brain Tumor Classification Training Script
BRISC 2025 Dataset - Multi-class Classification (Glioma, Meningioma, Pituitary, No Tumor)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime


class BrainTumorDataset(Dataset):
    """Custom Dataset for Brain Tumor Classification"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = os.path.join(root_dir, 'classification_task', split)
        self.transform = transform
        self.classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {split} set")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(img_size=224, augment=True):
    """
    Get data preprocessing and augmentation transforms
    
    Args:
        img_size (int): Target image size
        augment (bool): Whether to apply data augmentation
    
    Returns:
        dict: Dictionary with 'train' and 'test' transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return {'train': train_transform, 'test': test_transform}


def create_model(num_classes=4, model_name='resnet50', pretrained=True):
    """
    Create a CNN model for classification
    
    Args:
        num_classes (int): Number of output classes
        model_name (str): Model architecture ('resnet50', 'resnet18', 'efficientnet_b0')
        pretrained (bool): Use pretrained weights
    
    Returns:
        torch.nn.Module: CNN model
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}', 
                         'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}', 
                             'acc': f'{100.*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_training_history(history, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def main():
    # Configuration
    config = {
        'data_root': r'd:\Brain Tumor Project\brisc2025',
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'img_size': 224,
        'model_name': 'resnet50',  # resnet50, resnet18, efficientnet_b0
        'num_workers': 4,
        'augmentation': True,
        'patience': 10,  # Early stopping patience
        'output_dir': 'outputs/classification'
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
    transforms_dict = get_transforms(img_size=config['img_size'], 
                                     augment=config['augmentation'])
    
    # Create datasets
    print("\n=== Loading Datasets ===")
    train_dataset = BrainTumorDataset(
        config['data_root'], 
        split='train', 
        transform=transforms_dict['train']
    )
    
    test_dataset = BrainTumorDataset(
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
    model = create_model(
        num_classes=len(train_dataset.classes),
        model_name=config['model_name'],
        pretrained=True
    )
    model = model.to(device)
    print(f"Model: {config['model_name']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n=== Starting Training ===")
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(config['output_dir'], 'best_model.pth'))
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model for final evaluation
    print("\n=== Final Evaluation ===")
    checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(test_labels, test_preds, 
                                target_names=train_dataset.classes))
    
    # Save results
    with open(os.path.join(config['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(classification_report(test_labels, test_preds, 
                                     target_names=train_dataset.classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, train_dataset.classes,
                         os.path.join(config['output_dir'], 'confusion_matrix.png'))
    
    # Plot training history
    plot_training_history(history, 
                         os.path.join(config['output_dir'], 'training_history.png'))
    
    # Save training history
    with open(os.path.join(config['output_dir'], 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == '__main__':
    main()
