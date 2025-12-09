"""
Test Image Validation - Medical vs Non-Medical Images
"""
import requests
import os
from pathlib import Path

def test_image(image_path, description):
    """Test a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"File: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post('http://localhost:5000/api/classify', files=files)
        
        result = response.json()
        
        if 'error' in result:
            print(f"🚫 REJECTED: {result['error']}")
        else:
            print(f"✅ ACCEPTED")
            print(f"   Prediction: {result['predicted_class']}")
            print(f"   Confidence: {result['confidence']*100:.2f}%")
            print(f"   All Probabilities:")
            for cls, prob in result['probabilities'].items():
                print(f"      {cls:15s}: {prob*100:.2f}%")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("\n" + "="*70)
    print(" 🧪 TESTING IMAGE VALIDATION SYSTEM")
    print("="*70)
    
    # Test 1: Valid medical image (glioma)
    glioma_dir = Path("classification_task/test/glioma")
    if glioma_dir.exists():
        glioma_images = list(glioma_dir.glob("*.jpg"))
        if glioma_images:
            test_image(str(glioma_images[0]), "✅ Valid Medical Image - Glioma")
    
    # Test 2: Valid medical image (no tumor)
    notumor_dir = Path("classification_task/test/no_tumor")
    if notumor_dir.exists():
        notumor_images = list(notumor_dir.glob("*.jpg"))
        if notumor_images:
            test_image(str(notumor_images[0]), "✅ Valid Medical Image - No Tumor")
    
    # Test 3: Create a test non-medical image (colorful)
    print("\n" + "="*70)
    print("Creating test non-medical images...")
    print("="*70)
    
    try:
        from PIL import Image
        import numpy as np
        
        # Create a colorful gradient image
        test_dir = Path("test_images")
        test_dir.mkdir(exist_ok=True)
        
        # Test image 1: Colorful gradient
        color_img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                color_img[i, j] = [i, j, (i+j)//2]
        
        color_path = test_dir / "colorful_gradient.jpg"
        Image.fromarray(color_img).save(color_path)
        test_image(str(color_path), "❌ Non-Medical - Colorful Gradient")
        
        # Test image 2: Solid color
        solid_img = np.full((256, 256, 3), [255, 0, 0], dtype=np.uint8)
        solid_path = test_dir / "solid_red.jpg"
        Image.fromarray(solid_img).save(solid_path)
        test_image(str(solid_path), "❌ Non-Medical - Solid Red")
        
        # Test image 3: Very dark image
        dark_img = np.full((256, 256, 3), 5, dtype=np.uint8)
        dark_path = test_dir / "too_dark.jpg"
        Image.fromarray(dark_img).save(dark_path)
        test_image(str(dark_path), "❌ Non-Medical - Too Dark")
        
        # Test image 4: Very bright image
        bright_img = np.full((256, 256, 3), 250, dtype=np.uint8)
        bright_path = test_dir / "too_bright.jpg"
        Image.fromarray(bright_img).save(bright_path)
        test_image(str(bright_path), "❌ Non-Medical - Too Bright")
        
        # Test image 5: Wide aspect ratio
        wide_img = np.full((100, 400, 3), 128, dtype=np.uint8)
        wide_path = test_dir / "wide_aspect.jpg"
        Image.fromarray(wide_img).save(wide_path)
        test_image(str(wide_path), "❌ Non-Medical - Wide Aspect Ratio")
        
        print(f"\n✓ Test images created in: {test_dir.absolute()}")
        
    except Exception as e:
        print(f"Error creating test images: {e}")
    
    print("\n" + "="*70)
    print(" ✅ VALIDATION TESTING COMPLETE")
    print("="*70)
    print("\nSummary:")
    print("- Valid medical images should be ACCEPTED with predictions")
    print("- Non-medical images should be REJECTED with error messages")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
