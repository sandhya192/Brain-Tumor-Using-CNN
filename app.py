"""
Flask Web Application for Brain Tumor Classification and Segmentation
BRISC 2025 Dataset
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import base64
from werkzeug.utils import secure_filename
import json
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
classification_model = None
segmentation_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class labels
CLASS_LABELS = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def is_grayscale_dominant(image):
    """Check if image is predominantly grayscale (like medical scans)"""
    img_array = np.array(image)
    
    # Check if image has color channels
    if len(img_array.shape) < 3:
        return True
    
    # Calculate color variance between RGB channels
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Medical images typically have very low variance between channels
    rg_diff = np.abs(r.astype(float) - g.astype(float)).mean()
    rb_diff = np.abs(r.astype(float) - b.astype(float)).mean()
    gb_diff = np.abs(g.astype(float) - b.astype(float)).mean()
    
    avg_diff = (rg_diff + rb_diff + gb_diff) / 3
    
    # If average difference is less than 10, it's likely grayscale
    return avg_diff < 10


def validate_medical_image(image_path):
    """Validate if uploaded image is likely a medical brain scan"""
    try:
        image = Image.open(image_path)
        img_array = np.array(image.convert('RGB'))
        
        # Check 1: Image should be predominantly grayscale (medical scans)
        if not is_grayscale_dominant(image):
            return False, "Image appears to be a color photo, not a medical scan. Please upload a grayscale brain MRI."
        
        # Check 2: Aspect ratio check (brain scans are typically square-ish)
        width, height = image.size
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 3.0:
            return False, "Image aspect ratio unusual for brain scans. Please upload a valid MRI image."
        
        # Check 3: Size check - image should not be too small
        if width < 50 or height < 50:
            return False, "Image too small. Please upload a valid brain MRI scan."
        
        return True, "Valid medical image"
        
    except Exception as e:
        # If validation fails, log but allow - let model and confidence checks handle it
        print(f"[WARNING] Validation error for {image_path}: {str(e)}")
        return True, "Validation skipped due to error"


def load_classification_model(model_path='outputs/classification/best_model.pth'):
    """Load trained classification model"""
    try:
        # Create model
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(CLASS_LABELS))
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            print(f"✓ Classification model loaded from {model_path}")
            return model
        else:
            print(f"⚠ Classification model not found at {model_path}")
            return None
    except Exception as e:
        print(f"✗ Error loading classification model: {e}")
        return None


def load_segmentation_model(model_path='outputs/segmentation/best_model.pth'):
    """Load trained segmentation model"""
    try:
        from train_segmentation import UNet
        
        # Create model
        model = UNet(in_channels=3, out_channels=1)
        
        # Load weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            print(f"✓ Segmentation model loaded from {model_path}")
            return model
        else:
            print(f"⚠ Segmentation model not found at {model_path}")
            return None
    except Exception as e:
        print(f"✗ Error loading segmentation model: {e}")
        return None


def get_classification_transform():
    """Get preprocessing transform for classification"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_segmentation_transform():
    """Get preprocessing transform for segmentation"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def predict_classification(image_path):
    """Predict tumor class from image"""
    global classification_model
    
    if classification_model is None:
        return {'error': 'Classification model not loaded'}
    
    try:
        # Validate if image is a medical scan
        is_valid, validation_message = validate_medical_image(image_path)
        if not is_valid:
            return {'error': validation_message}
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        transform = get_classification_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = classification_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Additional confidence check - if all probabilities are too uniform, reject
        max_prob = probabilities.max().item()
        min_prob = probabilities.min().item()
        
        # If the difference between max and min probability is too small, image might not be medical
        if (max_prob - min_prob) < 0.2:
            return {'error': 'Model is uncertain about this image. Please upload a clear brain MRI scan.'}
        
        # If confidence is too low, reject
        if confidence < 0.5:
            return {'error': f'Model confidence too low ({confidence*100:.1f}%). This may not be a valid brain MRI scan.'}
        
        # Prepare results
        results = {
            'predicted_class': CLASS_LABELS[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                CLASS_LABELS[i]: float(probabilities[i].item()) 
                for i in range(len(CLASS_LABELS))
            }
        }
        
        return results
    
    except Exception as e:
        return {'error': str(e)}


def predict_segmentation(image_path):
    """Predict tumor segmentation mask"""
    global segmentation_model
    
    if segmentation_model is None:
        return {'error': 'Segmentation model not loaded'}
    
    try:
        # Validate if image is a medical scan
        is_valid, validation_message = validate_medical_image(image_path)
        if not is_valid:
            return {'error': validation_message}
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        transform = get_segmentation_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            mask_pred = segmentation_model(image_tensor)
            mask_pred = mask_pred.squeeze().cpu().numpy()
        
        # Convert to binary mask
        mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
        
        # Resize back to original size
        mask_image = Image.fromarray(mask_binary, mode='L')
        mask_image = mask_image.resize(original_size, Image.NEAREST)
        
        # Convert to base64 for display
        buffer = io.BytesIO()
        mask_image.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Calculate statistics
        tumor_pixels = np.sum(mask_binary > 0)
        total_pixels = mask_binary.size
        tumor_percentage = (tumor_pixels / total_pixels) * 100
        
        results = {
            'mask_base64': mask_base64,
            'tumor_percentage': float(tumor_percentage),
            'dice_score': None  # Can be calculated if ground truth is available
        }
        
        return results
    
    except Exception as e:
        return {'error': str(e)}


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/test-frontend')
def test_frontend():
    """Frontend connection test page"""
    return send_from_directory('.', 'test_frontend.html')


@app.route('/classify', methods=['GET'])
def classify_page():
    """Classification page"""
    return render_template('classify.html')


@app.route('/segment', methods=['GET'])
def segment_page():
    """Segmentation page"""
    return render_template('segment.html')


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API endpoint for classification"""
    print("\n[API] Classification request received")
    
    if 'file' not in request.files:
        print("[API] Error: No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        print("[API] Error: Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        print(f"[API] Error: Invalid file type - {file.filename}")
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"[API] File saved: {filename}")
        
        # Predict
        print(f"[API] Running classification...")
        results = predict_classification(filepath)
        
        if 'error' in results:
            print(f"[API] Prediction error: {results['error']}")
            return jsonify(results), 500
        
        print(f"[API] Prediction: {results['predicted_class']} ({results['confidence']*100:.2f}%)")
        
        # Add image path for display
        results['image_path'] = f'/uploads/{filename}'
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/segment', methods=['POST'])
def api_segment():
    """API endpoint for segmentation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict
        results = predict_segmentation(filepath)
        
        # Add image path for display
        results['image_path'] = f'/uploads/{filename}'
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-both', methods=['POST'])
def api_predict_both():
    """API endpoint for both classification and segmentation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict both
        classification_results = predict_classification(filepath)
        segmentation_results = predict_segmentation(filepath)
        
        results = {
            'image_path': f'/uploads/{filename}',
            'classification': classification_results,
            'segmentation': segmentation_results
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/classification_task/test/<tumor_type>/<filename>')
def dataset_image(tumor_type, filename):
    """Serve dataset images for testing"""
    return send_from_directory(f'classification_task/test/{tumor_type}', filename)


@app.route('/api/model-status')
def model_status():
    """Check which models are loaded"""
    return jsonify({
        'classification_loaded': classification_model is not None,
        'segmentation_loaded': segmentation_model is not None,
        'device': str(device)
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🧠 Brain Tumor Classification & Segmentation Web App")
    print("="*60)
    print(f"Device: {device}")
    print("\nLoading models...")
    
    # Load models
    classification_model = load_classification_model()
    segmentation_model = load_segmentation_model()
    
    print("\nModel Status:")
    print(f"  Classification: {'✓ Loaded' if classification_model else '✗ Not loaded'}")
    print(f"  Segmentation:   {'✓ Loaded' if segmentation_model else '✗ Not loaded'}")
    
    print("\n" + "="*60)
    print("Starting Flask server...")
    print("="*60)
    print("\nAccess the app at: http://localhost:5000")
    print("Press CTRL+C to stop the server\n")
    
    # Run without debug mode for stability
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
