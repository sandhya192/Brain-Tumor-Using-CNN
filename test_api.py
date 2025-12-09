"""
Test Classification API and Model Accuracy
"""
import requests
import json
from pathlib import Path
import time

# Test configuration
API_URL = "http://localhost:5000/api/classify"
TEST_DATA_DIR = Path("classification_task/test")

def test_single_image(image_path, expected_class):
    """Test classification on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path.name}")
    print(f"Expected: {expected_class}")
    
    try:
        # Upload and classify
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
        
        # Parse results
        result = response.json()
        
        predicted = result['predicted_class']
        confidence = result['confidence'] * 100
        
        print(f"Predicted: {predicted} ({confidence:.2f}% confidence)")
        
        # Show all probabilities
        print("\nAll probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls:12s}: {prob*100:6.2f}%")
        
        # Check if correct
        is_correct = predicted == expected_class
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"\nResult: {status}")
        
        return is_correct
    
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_class_accuracy(class_name, num_samples=5):
    """Test accuracy for a specific class"""
    print(f"\n{'#'*60}")
    print(f"# Testing class: {class_name.upper()}")
    print(f"{'#'*60}")
    
    class_dir = TEST_DATA_DIR / class_name
    if not class_dir.exists():
        print(f"❌ Directory not found: {class_dir}")
        return
    
    # Get test images
    images = list(class_dir.glob("*.jpg"))[:num_samples]
    
    if not images:
        print(f"❌ No images found in {class_dir}")
        return
    
    # Test each image
    correct = 0
    total = len(images)
    
    for img_path in images:
        if test_single_image(img_path, class_name):
            correct += 1
        time.sleep(0.1)  # Small delay between requests
    
    # Summary
    accuracy = (correct / total) * 100
    print(f"\n{'-'*60}")
    print(f"Class Accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"{'-'*60}")
    
    return correct, total


def main():
    """Run comprehensive tests"""
    print("\n" + "="*60)
    print("🧪 BRISC 2025 - Classification API Test Suite")
    print("="*60)
    
    # Test API connectivity
    print("\n1. Testing API connectivity...")
    try:
        response = requests.get("http://localhost:5000/api/model-status")
        if response.status_code == 200:
            status = response.json()
            print("✅ API is online")
            print(f"   Classification Model: {'✓ Loaded' if status['classification_loaded'] else '✗ Not Loaded'}")
            print(f"   Device: {status.get('device', 'Unknown')}")
        else:
            print("❌ API returned error:", response.status_code)
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("\n💡 Make sure Flask server is running:")
        print("   python app.py")
        return
    
    # Test each class
    classes = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
    total_correct = 0
    total_samples = 0
    
    for class_name in classes:
        correct, count = test_class_accuracy(class_name, num_samples=3)
        total_correct += correct
        total_samples += count
        time.sleep(0.5)
    
    # Overall summary
    print("\n" + "="*60)
    print("📊 OVERALL TEST SUMMARY")
    print("="*60)
    print(f"Total Correct: {total_correct}/{total_samples}")
    print(f"Overall Accuracy: {(total_correct/total_samples)*100:.2f}%")
    print("="*60)
    
    if total_correct == total_samples:
        print("\n🎉 Perfect! All predictions are correct!")
    elif (total_correct/total_samples) >= 0.9:
        print("\n✅ Great! Model is performing well!")
    else:
        print("\n⚠️  Some predictions are incorrect. This might be:")
        print("   1. Normal variation (no model is 100% accurate)")
        print("   2. Difficult test cases")
        print("   3. Model or preprocessing issues")


if __name__ == "__main__":
    main()
