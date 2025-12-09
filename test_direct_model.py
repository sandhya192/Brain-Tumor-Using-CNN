"""
Test direct model classification without GUI
"""
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

print("="*60)
print("Testing Direct Model Classification")
print("="*60)
print(f"Device: {device}")

# Load model
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_labels))

model_path = 'outputs/classification/best_model.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Model loaded from {model_path}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Test with multiple images from each class
test_cases = [
    ('classification_task/test/glioma/brisc2025_test_00001_gl_ax_t1.jpg', 'glioma'),
    ('classification_task/test/glioma/brisc2025_test_00002_gl_ax_t1.jpg', 'glioma'),
    ('classification_task/test/meningioma/brisc2025_test_00255_me_ax_t1.jpg', 'meningioma'),
    ('classification_task/test/pituitary/brisc2025_test_00701_pi_ax_t1.jpg', 'pituitary'),
    ('classification_task/test/no_tumor/brisc2025_test_00561_no_ax_t1.jpg', 'no_tumor'),
]

print("\n" + "="*60)
print("Testing Classification")
print("="*60)

correct = 0
total = 0

for img_path, expected in test_cases:
    if not os.path.exists(img_path):
        print(f"\n❌ File not found: {img_path}")
        continue
    
    # Load and preprocess
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    predicted = class_labels[predicted_class]
    is_correct = predicted == expected
    
    if is_correct:
        correct += 1
    total += 1
    
    status = "✅" if is_correct else "❌"
    
    print(f"\n{status} {os.path.basename(img_path)}")
    print(f"   Expected: {expected}")
    print(f"   Predicted: {predicted} ({confidence*100:.2f}%)")
    print(f"   All probabilities:")
    for i, label in enumerate(class_labels):
        prob = probabilities[i].item() * 100
        marker = "→" if i == predicted_class else " "
        print(f"   {marker} {label:12s}: {prob:6.2f}%")

print("\n" + "="*60)
print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
print("="*60)

if correct == total:
    print("\n🎉 Perfect! Model is classifying correctly!")
else:
    print("\n⚠️ Some misclassifications detected")
    print("This could indicate:")
    print("  1. Model needs retraining")
    print("  2. Test images are ambiguous")
    print("  3. Model weights not loaded correctly")
