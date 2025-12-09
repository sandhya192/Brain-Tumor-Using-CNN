"""
Brain Tumor Classification - Desktop Application
Direct model usage without API
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import os
from pathlib import Path

class BrainTumorClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BRISC 2025 - Brain Tumor Classifier")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.current_image_path = None
        self.current_image = None
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load trained classification model directly"""
        model_path = 'outputs/classification/best_model.pth'
        
        try:
            # Create model architecture
            self.model = models.resnet50(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.class_labels))
            
            # Load trained weights
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"✓ Model loaded successfully from {model_path}")
                print(f"✓ Using device: {self.device}")
            else:
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            print(f"Error loading model: {e}")
    
    def create_widgets(self):
        """Create GUI components"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#4F46E5', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🧠 Brain Tumor Classification System",
            font=('Arial', 24, 'bold'),
            bg='#4F46E5',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Model info
        info_frame = tk.Frame(self.root, bg='#f0f0f0')
        info_frame.pack(fill='x', padx=20, pady=10)
        
        device_label = tk.Label(
            info_frame,
            text=f"Device: {self.device} | Model: ResNet-50 | Classes: 4 (Glioma, Meningioma, Pituitary, No Tumor)",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#666'
        )
        device_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Image display
        left_frame = tk.LabelFrame(
            main_frame,
            text="Input Image",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image display
        self.image_label = tk.Label(
            left_frame,
            text="No image loaded\n\nClick 'Select Image' to begin",
            font=('Arial', 14),
            bg='white',
            fg='#999',
            width=40,
            height=20
        )
        self.image_label.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='white')
        button_frame.pack(pady=10)
        
        self.select_btn = tk.Button(
            button_frame,
            text="📁 Select Image",
            command=self.select_image,
            font=('Arial', 12, 'bold'),
            bg='#4F46E5',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.select_btn.pack(side='left', padx=5)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="🔬 Classify",
            command=self.classify_image,
            font=('Arial', 12, 'bold'),
            bg='#10B981',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2',
            state='disabled'
        )
        self.classify_btn.pack(side='left', padx=5)
        
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_results,
            font=('Arial', 12, 'bold'),
            bg='#EF4444',
            fg='white',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.clear_btn.pack(side='left', padx=5)
        
        # Right panel - Results
        right_frame = tk.LabelFrame(
            main_frame,
            text="Classification Results",
            font=('Arial', 12, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Prediction result
        self.result_frame = tk.Frame(right_frame, bg='white')
        self.result_frame.pack(fill='x', pady=10)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Awaiting classification...",
            font=('Arial', 14),
            bg='white',
            fg='#999'
        )
        self.result_label.pack(pady=20)
        
        # Probabilities
        prob_label = tk.Label(
            right_frame,
            text="Class Probabilities:",
            font=('Arial', 11, 'bold'),
            bg='white',
            anchor='w'
        )
        prob_label.pack(fill='x', pady=(10, 5))
        
        # Progress bars for each class
        self.prob_frame = tk.Frame(right_frame, bg='white')
        self.prob_frame.pack(fill='both', expand=True, pady=10)
        
        self.prob_bars = {}
        self.prob_labels = {}
        
        for i, class_name in enumerate(self.class_labels):
            # Class name label
            class_frame = tk.Frame(self.prob_frame, bg='white')
            class_frame.pack(fill='x', pady=5)
            
            label = tk.Label(
                class_frame,
                text=class_name.replace('_', ' ').title(),
                font=('Arial', 10),
                bg='white',
                width=15,
                anchor='w'
            )
            label.pack(side='left')
            
            # Progress bar
            progress = ttk.Progressbar(
                class_frame,
                length=250,
                mode='determinate',
                maximum=100
            )
            progress.pack(side='left', padx=5)
            self.prob_bars[class_name] = progress
            
            # Percentage label
            percent_label = tk.Label(
                class_frame,
                text="0.00%",
                font=('Arial', 10, 'bold'),
                bg='white',
                width=8
            )
            percent_label.pack(side='left')
            self.prob_labels[class_name] = percent_label
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=('Arial', 9),
            bg='#e0e0e0',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def select_image(self):
        """Open file dialog to select image"""
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.display_image(file_path)
                self.classify_btn.config(state='normal')
                self.status_bar.config(text=f"Loaded: {Path(file_path).name}")
                self.clear_results()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image_path):
        """Display selected image"""
        try:
            # Load and resize image for display
            image = Image.open(image_path)
            image.thumbnail((400, 400), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def get_transform(self):
        """Get preprocessing transform (same as training)"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def validate_medical_image(self, image):
        """Quick validation if image looks like medical scan"""
        img_array = np.array(image.convert('RGB'))
        
        # Check if grayscale-dominant (medical scans)
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        rg_diff = np.abs(r.astype(float) - g.astype(float)).mean()
        rb_diff = np.abs(r.astype(float) - b.astype(float)).mean()
        gb_diff = np.abs(g.astype(float) - b.astype(float)).mean()
        avg_diff = (rg_diff + rb_diff + gb_diff) / 3
        
        if avg_diff > 10:
            return False, "Image appears to be a color photo. Please use grayscale brain MRI scans."
        
        return True, "Valid"
    
    def classify_image(self):
        """Classify the selected image using loaded model"""
        if not self.current_image_path or self.model is None:
            return
        
        try:
            self.status_bar.config(text="Classifying...")
            self.root.update()
            
            # Load image
            image = Image.open(self.current_image_path).convert('RGB')
            
            # Validate
            is_valid, msg = self.validate_medical_image(image)
            if not is_valid:
                messagebox.showwarning("Invalid Image", msg)
                self.status_bar.config(text="Ready")
                return
            
            # Preprocess
            transform = self.get_transform()
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Predict (DIRECTLY using the model, no API)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            # Additional validation - check probability distribution
            max_prob = probabilities.max().item()
            min_prob = probabilities.min().item()
            prob_std = probabilities.std().item()
            
            # If probabilities are too uniform, likely not a medical image
            if (max_prob - min_prob) < 0.3 or prob_std < 0.1:
                messagebox.showerror(
                    "Invalid Image",
                    f"This doesn't appear to be a valid brain MRI scan.\n\n"
                    f"Model is uncertain (probabilities too uniform).\n"
                    f"Please upload a proper brain MRI image."
                )
                self.status_bar.config(text="Invalid image - classification rejected")
                return
            
            # Check confidence
            if confidence < 0.6:
                result = messagebox.askyesno(
                    "Low Confidence",
                    f"Model confidence is only {confidence*100:.1f}%.\n\n"
                    f"This may not be a valid brain MRI scan.\n\n"
                    f"Do you want to see the results anyway?"
                )
                if not result:
                    self.status_bar.config(text="Classification cancelled")
                    return
            
            # Display results
            self.display_results(predicted_class, confidence, probabilities)
            
            self.status_bar.config(
                text=f"Classification complete - {self.class_labels[predicted_class].title()} ({confidence*100:.2f}%)"
            )
            
        except Exception as e:
            messagebox.showerror("Classification Error", f"Failed to classify image: {str(e)}")
            self.status_bar.config(text="Error")
            print(f"Classification error: {e}")
    
    def display_results(self, predicted_class, confidence, probabilities):
        """Display classification results"""
        # Update main result
        class_name = self.class_labels[predicted_class]
        
        self.result_label.config(
            text=f"Predicted: {class_name.replace('_', ' ').upper()}\n\nConfidence: {confidence*100:.2f}%",
            font=('Arial', 18, 'bold'),
            fg='#10B981' if confidence > 0.8 else '#F59E0B'
        )
        
        # Update probability bars
        for i, class_name in enumerate(self.class_labels):
            prob = probabilities[i].item() * 100
            self.prob_bars[class_name]['value'] = prob
            self.prob_labels[class_name].config(
                text=f"{prob:.2f}%",
                fg='#10B981' if i == predicted_class else '#666'
            )
    
    def clear_results(self):
        """Clear results and reset"""
        self.result_label.config(
            text="Awaiting classification...",
            font=('Arial', 14),
            fg='#999'
        )
        
        for class_name in self.class_labels:
            self.prob_bars[class_name]['value'] = 0
            self.prob_labels[class_name].config(text="0.00%", fg='#666')
        
        if self.current_image_path:
            self.status_bar.config(text=f"Loaded: {Path(self.current_image_path).name}")
        else:
            self.status_bar.config(text="Ready")


def main():
    """Run the application"""
    root = tk.Tk()
    app = BrainTumorClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
