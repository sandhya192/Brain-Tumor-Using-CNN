"""
Diagnostic: Verify Model Class Mappings and Test Classification (ResNet-50)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np

print("="*80)
print("MODEL DIAGNOSTIC - Class Label Verification (ResNet-50)")
print("="*80)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'outputs/classification/best_model.pth'

print(f"\n[1] Loading model from: {model_path}")
print(f"[2] Device: {device}")

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"[3] Model loaded successfully")
print(f"[4] Output classes: {model.fc.out_features}")

# Check what class labels were used during training
print("\n" + "="*80)
print("CHECKING TRAINING CONFIGURATION")
print("="*80)

# Expected class order from dataset folders
expected_order = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
print(f"\nExpected class order (alphabetical): {expected_order}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test with known images from dataset
print("\n" + "="*80)
print("TESTING WITH DATASET IMAGES")
print("="*80)

test_cases = [
    ('classification_task/test/glioma', 'glioma'),
    ('classification_task/test/meningioma', 'meningioma'),
    ('classification_task/test/no_tumor', 'no_tumor'),
    ('classification_task/test/pituitary', 'pituitary')
]

results = []

for folder, true_label in test_cases:
    if os.path.exists(folder):
        # Get first image
        images = [f for f in os.listdir(folder) if f.endswith('.jpg')]
        if images:
            img_path = os.path.join(folder, images[0])
            
            # Load and classify
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
            
            predicted_label = expected_order[pred_idx]
            is_correct = predicted_label == true_label
            
            print(f"\n{true_label.upper():12s} sample: {images[0]}")
            print(f"  Raw output index: {pred_idx}")
            print(f"  Predicted: {predicted_label.upper()} ({confidence*100:.2f}%)")
            print(f"  Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print(f"  All probabilities:")
            for i, label in enumerate(expected_order):
                marker = " <--" if i == pred_idx else ""
                print(f"    [{i}] {label:12s}: {probs[i].item()*100:6.2f}%{marker}")
            
            results.append(is_correct)

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
correct = sum(results)
total = len(results)
print(f"\nCorrect classifications: {correct}/{total}")
print(f"Accuracy: {(correct/total)*100:.1f}%")

if correct == total:
    print("\n✓ Model is working correctly!")
    print("✓ Class labels are mapped properly")
else:
    print("\n✗ Model has incorrect class mappings!")
    print("\nPossible issues:")
    print("  1. Class order during training was different")
    print("  2. Model checkpoint has wrong class ordering")
    
# Additional check - verify what index 2 corresponds to
print("\n" + "="*80)
print("CLASS INDEX MAPPING")
print("="*80)
print("\nCurrent mapping being used:")
for i, label in enumerate(expected_order):
    print(f"  Index {i} -> {label.upper()}")

print("\n" + "="*80)
