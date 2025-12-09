"""
Multi-Slice Brain MRI Classifier
Handles images with multiple brain scan slices
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import os
from pathlib import Path

class MultiSliceBrainTumorClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("BRISC 2025 - Multi-Slice Brain Tumor Classifier")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Model variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.current_image_path = None
        
        # Load model
        self.load_model()
        
        # Create GUI
        self.create_widgets()
        
    def load_model(self):
        """Load trained classification model"""
        model_path = 'outputs/classification/best_model.pth'
        
        try:
            self.model = models.resnet50(pretrained=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, len(self.class_labels))
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                print(f"Model loaded successfully from {model_path}")
                print(f"Using device: {self.device}")
            else:
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def create_widgets(self):
        """Create GUI components"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='#4F46E5', height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="Brain Tumor Multi-Slice Classifier",
            font=('Arial', 20, 'bold'),
            bg='#4F46E5',
            fg='white'
        )
        title_label.pack(pady=25)
        
        # Info bar
        info_frame = tk.Frame(self.root, bg='#f0f0f0')
        info_frame.pack(fill='x', padx=20, pady=5)
        
        info_label = tk.Label(
            info_frame,
            text=f"Device: {self.device} | Handles both single and multi-view MRI scans",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#666'
        )
        info_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Image
        left_frame = tk.LabelFrame(
            main_frame,
            text="Input Image",
            font=('Arial', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.image_label = tk.Label(
            left_frame,
            text="No image loaded\n\nClick 'Select Image' to begin",
            font=('Arial', 12),
            bg='white',
            fg='#999'
        )
        self.image_label.pack(pady=20, fill='both', expand=True)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='white')
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text="Select Image",
            command=self.select_image,
            font=('Arial', 11, 'bold'),
            bg='#4F46E5',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="Classify",
            command=self.classify_image,
            font=('Arial', 11, 'bold'),
            bg='#10B981',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2',
            state='disabled'
        )
        self.classify_btn.pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_results,
            font=('Arial', 11, 'bold'),
            bg='#EF4444',
            fg='white',
            padx=15,
            pady=8,
            cursor='hand2'
        ).pack(side='left', padx=5)
        
        # Right panel - Results
        right_frame = tk.LabelFrame(
            main_frame,
            text="Classification Results",
            font=('Arial', 11, 'bold'),
            bg='white',
            padx=10,
            pady=10
        )
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            right_frame,
            font=('Consolas', 10),
            bg='#f9f9f9',
            wrap=tk.WORD,
            height=30
        )
        self.results_text.pack(fill='both', expand=True, pady=5)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready - Supports single slice and multi-view MRI scans",
            font=('Arial', 9),
            bg='#e0e0e0',
            anchor='w',
            padx=10
        )
        self.status_bar.pack(side='bottom', fill='x')
    
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
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
        """Display the selected image"""
        try:
            image = Image.open(image_path)
            image.thumbnail((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")
    
    def detect_grid_layout(self, image):
        """Detect if image contains multiple brain scans in a grid"""
        width, height = image.size
        img_array = np.array(image.convert('L'))
        
        # Check aspect ratio - multi-view images are usually not square
        aspect_ratio = width / height
        
        # If roughly 4:3 or 3:4, likely a multi-view grid
        if 1.2 < aspect_ratio < 1.5 or 0.67 < aspect_ratio < 0.83:
            # Estimate grid size (common: 3x4, 4x3, 4x4)
            if width > height:
                # Landscape - probably 4 columns x 3 rows
                return True, 4, 3
            else:
                # Portrait - probably 3 columns x 4 rows
                return True, 3, 4
        
        # Check if it's a 2x2, 2x3, etc.
        if aspect_ratio > 1.5:  # Wide image
            return True, 4, 2
        elif aspect_ratio < 0.67:  # Tall image
            return True, 2, 4
        
        # If square-ish, assume single slice
        return False, 1, 1
    
    def extract_slices(self, image, rows, cols):
        """Extract individual slices from grid"""
        width, height = image.size
        slice_width = width // cols
        slice_height = height // rows
        
        slices = []
        for row in range(rows):
            for col in range(cols):
                left = col * slice_width
                top = row * slice_height
                right = left + slice_width
                bottom = top + slice_height
                
                slice_img = image.crop((left, top, right, bottom))
                slices.append((slice_img, row, col))
        
        return slices
    
    def get_transform(self):
        """Preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def classify_single_slice(self, slice_img):
        """Classify a single brain scan slice"""
        try:
            # Convert to RGB
            slice_rgb = slice_img.convert('RGB')
            
            # Transform
            transform = self.get_transform()
            image_tensor = transform(slice_rgb).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
            
            return {
                'class': self.class_labels[predicted_idx],
                'confidence': confidence,
                'probabilities': {
                    self.class_labels[i]: probabilities[i].item() 
                    for i in range(len(self.class_labels))
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def classify_image(self):
        """Classify the loaded image"""
        if not self.current_image_path or self.model is None:
            return
        
        try:
            self.status_bar.config(text="Analyzing image...")
            self.root.update()
            
            # Load image
            image = Image.open(self.current_image_path)
            
            # Detect layout
            is_grid, rows, cols = self.detect_grid_layout(image)
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "="*60 + "\n")
            self.results_text.insert(tk.END, "CLASSIFICATION RESULTS\n")
            self.results_text.insert(tk.END, "="*60 + "\n\n")
            
            if is_grid:
                # Multi-slice image
                self.results_text.insert(tk.END, f"Detected: Multi-view grid ({rows}x{cols} layout)\n")
                self.results_text.insert(tk.END, f"Total slices: {rows * cols}\n\n")
                self.results_text.insert(tk.END, "-"*60 + "\n\n")
                
                # Extract and classify each slice
                slices = self.extract_slices(image, rows, cols)
                
                # Count findings
                findings = {'glioma': 0, 'meningioma': 0, 'pituitary': 0, 'no_tumor': 0}
                
                for idx, (slice_img, row, col) in enumerate(slices, 1):
                    result = self.classify_single_slice(slice_img)
                    
                    if 'error' not in result:
                        predicted_class = result['class']
                        confidence = result['confidence']
                        findings[predicted_class] += 1
                        
                        self.results_text.insert(tk.END, f"Slice {idx} (Row {row+1}, Col {col+1}):\n")
                        self.results_text.insert(tk.END, f"  Predicted: {predicted_class.upper()}\n")
                        self.results_text.insert(tk.END, f"  Confidence: {confidence*100:.2f}%\n\n")
                    
                    self.root.update()
                
                # Summary
                self.results_text.insert(tk.END, "\n" + "="*60 + "\n")
                self.results_text.insert(tk.END, "SUMMARY\n")
                self.results_text.insert(tk.END, "="*60 + "\n\n")
                
                for tumor_type, count in findings.items():
                    if count > 0:
                        percentage = (count / (rows * cols)) * 100
                        self.results_text.insert(tk.END, f"{tumor_type.upper()}: {count} slices ({percentage:.1f}%)\n")
                
                # Overall assessment
                max_finding = max(findings, key=findings.get)
                if findings[max_finding] > 0:
                    self.results_text.insert(tk.END, f"\nPrimary Finding: {max_finding.upper()}\n")
                
            else:
                # Single slice
                self.results_text.insert(tk.END, "Detected: Single brain scan slice\n\n")
                self.results_text.insert(tk.END, "-"*60 + "\n\n")
                
                result = self.classify_single_slice(image)
                
                if 'error' in result:
                    self.results_text.insert(tk.END, f"Error: {result['error']}\n")
                else:
                    self.results_text.insert(tk.END, f"Predicted: {result['class'].upper()}\n")
                    self.results_text.insert(tk.END, f"Confidence: {result['confidence']*100:.2f}%\n\n")
                    self.results_text.insert(tk.END, "Probabilities:\n")
                    for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                        self.results_text.insert(tk.END, f"  {cls:12s}: {prob*100:6.2f}%\n")
            
            self.status_bar.config(text="Classification complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.status_bar.config(text="Error")
    
    def clear_results(self):
        """Clear results"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Awaiting classification...\n\n")
        self.results_text.insert(tk.END, "Upload an image and click 'Classify'")

def main():
    root = tk.Tk()
    app = MultiSliceBrainTumorClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()
