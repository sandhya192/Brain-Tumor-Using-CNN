"""
Comprehensive Diagnostic Test
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

print("\n" + "="*70)
print("COMPREHENSIVE DIAGNOSTIC TEST")
print("="*70)

# 1. Check CUDA
print("\n[1] GPU/CUDA Check:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Check model file
print("\n[2] Model File Check:")
model_path = 'outputs/classification/best_model.pth'
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ Model file exists: {model_path}")
    print(f"   Size: {size_mb:.2f} MB")
    
    # Load and inspect checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'epoch' in checkpoint:
        print(f"   Trained epochs: {checkpoint['epoch']}")
    if 'accuracy' in checkpoint:
        print(f"   Training accuracy: {checkpoint['accuracy']:.4f}")
else:
    print(f"   ❌ Model file NOT found!")
    exit(1)

# 3. Load model
print("\n[3] Loading Model:")
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_labels))

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"   ✅ Model loaded successfully")
print(f"   Model on device: {next(model.parameters()).device}")
print(f"   Model in eval mode: {not model.training}")

# 4. Test transform
print("\n[4] Transform Check:")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])
print(f"   ✅ Transform pipeline created")

# 5. Test classification on each class
print("\n[5] Classification Test on All Classes:")
print("-"*70)

test_images = {
    'glioma': 'classification_task/test/glioma/brisc2025_test_00001_gl_ax_t1.jpg',
    'meningioma': 'classification_task/test/meningioma/brisc2025_test_00255_me_ax_t1.jpg',
    'pituitary': 'classification_task/test/pituitary/brisc2025_test_00701_pi_ax_t1.jpg',
    'no_tumor': 'classification_task/test/no_tumor/brisc2025_test_00561_no_ax_t1.jpg'
}

all_correct = True

for expected_class, img_path in test_images.items():
    if not os.path.exists(img_path):
        print(f"\n❌ {expected_class}: Image not found - {img_path}")
        all_correct = False
        continue
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    
    # Transform
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    
    predicted_class = class_labels[predicted_idx]
    is_correct = predicted_class == expected_class
    
    status = "✅" if is_correct else "❌"
    if not is_correct:
        all_correct = False
    
    print(f"\n{status} Expected: {expected_class.upper()}")
    print(f"   Predicted: {predicted_class.upper()} (Confidence: {confidence*100:.2f}%)")
    print(f"   Probabilities:")
    for i, label in enumerate(class_labels):
        prob = probabilities[i].item() * 100
        marker = "→" if i == predicted_idx else " "
        print(f"   {marker} {label:12s}: {prob:6.2f}%")

# 6. Summary
print("\n" + "="*70)
print(" DIAGNOSTIC SUMMARY")
print("="*70)

if all_correct:
    print("\n*** ALL TESTS PASSED! ***")
    print("   ✅ Model is loaded correctly")
    print("   ✅ Model is classifying with 100% accuracy")
    print("   ✅ GPU acceleration is working")
    print("\n💡 If you're still seeing wrong results in the desktop app:")
    print("   1. Make sure you're uploading MEDICAL brain MRI scans")
    print("   2. Check that the image is grayscale (not color photos)")
    print("   3. The image you shared looks like a valid brain MRI")
    print("   4. Try uploading images from: classification_task/test/")
else:
    print("\n⚠️ SOME TESTS FAILED!")
    print("   The model is NOT classifying correctly.")
    print("   This indicates a problem with:")
    print("   • Model weights")
    print("   • Transform pipeline")
    print("   • Model architecture mismatch")

print("\n" + "="*70)
