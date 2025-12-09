"""
Test multi-view image classification
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import sys

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Load model
print("Loading model...")
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_labels))

checkpoint = torch.load('outputs/classification/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"Model loaded on {device}\n")

# Get transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_slice(img):
    """Classify a single slice"""
    img_rgb = img.convert('RGB')
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        conf = probs[pred_idx].item()
    
    return class_labels[pred_idx], conf, probs

# Ask user for image path
print("=" * 70)
print("MULTI-VIEW MRI CLASSIFIER TEST")
print("=" * 70)
print("\nPlease provide the path to your multi-view brain MRI image.")
print("(Drag and drop the file here, or paste the path)\n")

image_path = input("Image path: ").strip().strip('"')

try:
    # Load image
    img = Image.open(image_path)
    width, height = img.size
    print(f"\n✓ Image loaded: {width}x{height} pixels\n")
    
    # Detect layout
    aspect_ratio = width / height
    print(f"Aspect ratio: {aspect_ratio:.2f}")
    
    # Determine grid size
    if 1.2 < aspect_ratio < 1.5:  # Landscape 4x3
        rows, cols = 3, 4
        print(f"Detected layout: {rows}x{cols} grid (12 slices)\n")
    elif 0.67 < aspect_ratio < 0.83:  # Portrait 3x4
        rows, cols = 4, 3
        print(f"Detected layout: {rows}x{cols} grid (12 slices)\n")
    else:
        rows, cols = 1, 1
        print("Detected layout: Single slice\n")
    
    # Extract and classify slices
    slice_width = width // cols
    slice_height = height // rows
    
    print("=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70 + "\n")
    
    findings = {'glioma': 0, 'meningioma': 0, 'pituitary': 0, 'no_tumor': 0}
    
    for row in range(rows):
        for col in range(cols):
            slice_num = row * cols + col + 1
            
            # Extract slice
            left = col * slice_width
            top = row * slice_height
            right = left + slice_width
            bottom = top + slice_height
            
            slice_img = img.crop((left, top, right, bottom))
            
            # Classify
            pred_class, confidence, probs = classify_slice(slice_img)
            findings[pred_class] += 1
            
            print(f"Slice {slice_num:2d} (Row {row+1}, Col {col+1}): {pred_class.upper():12s} ({confidence*100:6.2f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70 + "\n")
    
    total_slices = rows * cols
    for tumor_type, count in sorted(findings.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / total_slices) * 100
            print(f"{tumor_type.upper():12s}: {count:2d} slices ({percentage:5.1f}%)")
    
    # Overall finding
    max_finding = max(findings, key=findings.get)
    print(f"\n>>> PRIMARY FINDING: {max_finding.upper()}")
    
    if findings['glioma'] > 0 or findings['meningioma'] > 0 or findings['pituitary'] > 0:
        tumor_slices = findings['glioma'] + findings['meningioma'] + findings['pituitary']
        print(f">>> TUMOR DETECTED in {tumor_slices}/{total_slices} slices")
    else:
        print(f">>> NO TUMOR DETECTED")
    
except FileNotFoundError:
    print(f"\n✗ ERROR: File not found: {image_path}")
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
