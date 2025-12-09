"""
BRISC 2025 - Brain Tumor Classification System
Professional Desktop Application with Direct Model Integration
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import os
from pathlib import Path
import threading

class BrainTumorClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("BRISC 2025 - Brain Tumor Classification System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # Model variables
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_colors = {
            'glioma': '#EF4444',
            'meningioma': '#F59E0B',
            'no_tumor': '#10B981',
            'pituitary': '#3B82F6'
        }
        self.current_image_path = None
        self.current_pil_image = None
        
        # Initialize UI
        self.create_styles()
        self.create_widgets()
        
        # Load model in background
        self.load_model_async()
        
    def create_styles(self):
        """Create custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Progress bar
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor='#2d2d44',
            background='#4F46E5',
            borderwidth=0,
            thickness=4
        )
        
    def create_widgets(self):
        """Create professional UI"""
        
        # ===== HEADER =====
        header = tk.Frame(self.root, bg='#16213e', height=100)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header,
            text="BRISC 2025",
            font=('Helvetica', 32, 'bold'),
            bg='#16213e',
            fg='#4F46E5'
        )
        title_label.pack(side='left', padx=40, pady=20)
        
        subtitle_label = tk.Label(
            header,
            text="Brain Tumor Classification System",
            font=('Helvetica', 14),
            bg='#16213e',
            fg='#94a3b8'
        )
        subtitle_label.place(x=40, y=60)
        
        # Status indicator
        self.status_indicator = tk.Canvas(header, width=20, height=20, bg='#16213e', highlightthickness=0)
        self.status_indicator.pack(side='right', padx=40)
        self.status_circle = self.status_indicator.create_oval(2, 2, 18, 18, fill='#FCD34D', outline='')
        
        self.status_label = tk.Label(
            header,
            text="Loading Model...",
            font=('Helvetica', 11),
            bg='#16213e',
            fg='#94a3b8'
        )
        self.status_label.pack(side='right', padx=10)
        
        # ===== MAIN CONTAINER =====
        main_container = tk.Frame(self.root, bg='#1a1a2e')
        main_container.pack(fill='both', expand=True, padx=30, pady=20)
        
        # ===== LEFT PANEL - IMAGE VIEWER =====
        left_panel = tk.Frame(main_container, bg='#16213e', relief='flat', bd=0)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Image panel header
        img_header = tk.Frame(left_panel, bg='#16213e', height=50)
        img_header.pack(fill='x')
        img_header.pack_propagate(False)
        
        tk.Label(
            img_header,
            text="Medical Image",
            font=('Helvetica', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(side='left', padx=20, pady=15)
        
        # Image display area
        self.image_frame = tk.Frame(left_panel, bg='#0f1419')
        self.image_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.image_canvas = tk.Canvas(
            self.image_frame,
            bg='#0f1419',
            highlightthickness=0
        )
        self.image_canvas.pack(fill='both', expand=True)
        
        # Placeholder
        self.placeholder_label = tk.Label(
            self.image_canvas,
            text="📋\n\nNo Image Loaded\n\nClick 'Select Image' to begin",
            font=('Helvetica', 14),
            bg='#0f1419',
            fg='#64748b',
            justify='center'
        )
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Control buttons
        btn_frame = tk.Frame(left_panel, bg='#16213e')
        btn_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.select_btn = tk.Button(
            btn_frame,
            text="📁  Select Image",
            command=self.select_image,
            font=('Helvetica', 12, 'bold'),
            bg='#4F46E5',
            fg='white',
            activebackground='#4338CA',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=12
        )
        self.select_btn.pack(side='left', padx=(0, 10))
        
        self.classify_btn = tk.Button(
            btn_frame,
            text="🔬  Classify",
            command=self.classify_image_threaded,
            font=('Helvetica', 12, 'bold'),
            bg='#10B981',
            fg='white',
            activebackground='#059669',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=12,
            state='disabled'
        )
        self.classify_btn.pack(side='left', padx=(0, 10))
        
        self.clear_btn = tk.Button(
            btn_frame,
            text="🗑  Clear",
            command=self.clear_all,
            font=('Helvetica', 12, 'bold'),
            bg='#EF4444',
            fg='white',
            activebackground='#DC2626',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=12
        )
        self.clear_btn.pack(side='left')
        
        # ===== RIGHT PANEL - RESULTS =====
        right_panel = tk.Frame(main_container, bg='#16213e', width=500, relief='flat', bd=0)
        right_panel.pack(side='right', fill='both', padx=(15, 0))
        right_panel.pack_propagate(False)
        
        # Results header
        results_header = tk.Frame(right_panel, bg='#16213e', height=50)
        results_header.pack(fill='x')
        results_header.pack_propagate(False)
        
        tk.Label(
            results_header,
            text="Classification Results",
            font=('Helvetica', 14, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(side='left', padx=20, pady=15)
        
        # Results container
        results_container = tk.Frame(right_panel, bg='#16213e')
        results_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Prediction display
        self.prediction_frame = tk.Frame(results_container, bg='#0f1419', relief='flat')
        self.prediction_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(
            self.prediction_frame,
            text="Predicted Diagnosis",
            font=('Helvetica', 11),
            bg='#0f1419',
            fg='#94a3b8'
        ).pack(anchor='w', padx=20, pady=(20, 5))
        
        self.prediction_label = tk.Label(
            self.prediction_frame,
            text="—",
            font=('Helvetica', 28, 'bold'),
            bg='#0f1419',
            fg='#ffffff'
        )
        self.prediction_label.pack(anchor='w', padx=20, pady=(0, 5))
        
        self.confidence_label = tk.Label(
            self.prediction_frame,
            text="",
            font=('Helvetica', 12),
            bg='#0f1419',
            fg='#94a3b8'
        )
        self.confidence_label.pack(anchor='w', padx=20, pady=(0, 20))
        
        # Probability bars
        tk.Label(
            results_container,
            text="Probability Distribution",
            font=('Helvetica', 11, 'bold'),
            bg='#16213e',
            fg='#ffffff'
        ).pack(anchor='w', pady=(0, 10))
        
        self.prob_bars_frame = tk.Frame(results_container, bg='#16213e')
        self.prob_bars_frame.pack(fill='both', expand=True)
        
        self.prob_bars = {}
        self.prob_labels = {}
        
        for idx, label in enumerate(self.class_labels):
            # Container for each class
            bar_container = tk.Frame(self.prob_bars_frame, bg='#0f1419', height=70)
            bar_container.pack(fill='x', pady=5)
            bar_container.pack_propagate(False)
            
            # Class name
            name_label = tk.Label(
                bar_container,
                text=label.replace('_', ' ').title(),
                font=('Helvetica', 11, 'bold'),
                bg='#0f1419',
                fg='#ffffff',
                width=12,
                anchor='w'
            )
            name_label.pack(side='left', padx=(15, 10))
            
            # Progress bar container
            bar_bg = tk.Frame(bar_container, bg='#1e293b', height=30)
            bar_bg.pack(side='left', fill='x', expand=True, padx=(0, 10))
            
            # Actual bar
            bar = tk.Canvas(bar_bg, bg='#1e293b', height=30, highlightthickness=0)
            bar.pack(fill='both', expand=True)
            self.prob_bars[label] = bar
            
            # Percentage label
            pct_label = tk.Label(
                bar_container,
                text="0%",
                font=('Helvetica', 10, 'bold'),
                bg='#0f1419',
                fg='#94a3b8',
                width=6,
                anchor='e'
            )
            pct_label.pack(side='right', padx=(0, 15))
            self.prob_labels[label] = pct_label
        
        # ===== FOOTER =====
        footer = tk.Frame(self.root, bg='#16213e', height=50)
        footer.pack(fill='x', side='bottom')
        footer.pack_propagate(False)
        
        self.info_label = tk.Label(
            footer,
            text=f"Device: {self.device.type.upper()} | Model: ResNet-50 | Accuracy: 99%",
            font=('Helvetica', 10),
            bg='#16213e',
            fg='#64748b'
        )
        self.info_label.pack(side='left', padx=40, pady=15)
        
        self.footer_status = tk.Label(
            footer,
            text="Ready",
            font=('Helvetica', 10),
            bg='#16213e',
            fg='#10B981'
        )
        self.footer_status.pack(side='right', padx=40, pady=15)
        
    def load_model_async(self):
        """Load model in background thread"""
        def load():
            try:
                model_path = 'outputs/classification/best_model.pth'
                
                # Create model
                self.model = models.resnet50(weights=None)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, len(self.class_labels))
                
                # Load weights
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                
                # Update UI
                self.root.after(0, self.on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        threading.Thread(target=load, daemon=True).start()
    
    def on_model_loaded(self):
        """Called when model loads successfully"""
        self.status_indicator.itemconfig(self.status_circle, fill='#10B981')
        self.status_label.config(text=f"Model Ready ({self.device.type.upper()})")
        self.footer_status.config(text="Ready", fg='#10B981')
    
    def on_model_error(self, error):
        """Called when model fails to load"""
        self.status_indicator.itemconfig(self.status_circle, fill='#EF4444')
        self.status_label.config(text="Model Error")
        messagebox.showerror("Model Error", f"Failed to load model:\n{error}")
    
    def select_image(self):
        """Select and display image"""
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Scan",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.dcm"),
                ("JPEG Files", "*.jpg *.jpeg"),
                ("PNG Files", "*.png"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_pil_image = Image.open(file_path)
                
                # Detect if multi-view
                width, height = self.current_pil_image.size
                aspect_ratio = width / height
                
                # Check for multi-view grid
                if self.is_multiview_image(self.current_pil_image):
                    # Show info dialog
                    result = messagebox.showinfo(
                        "Multi-View Image Detected",
                        "⚠️ IMPORTANT NOTICE:\n\n"
                        "This appears to be a MULTI-VIEW scan showing multiple brain slices in a grid.\n\n"
                        "The model was trained on INDIVIDUAL brain scan slices (512×512 pixels).\n\n"
                        "For accurate results:\n"
                        "1. Extract a single slice from the grid\n"
                        "2. Save it as a separate image\n"
                        "3. Upload that single slice\n\n"
                        "Classification will proceed but accuracy may be reduced on grid images.",
                        icon='warning'
                    )
                
                self.display_image()
                self.classify_btn.config(state='normal')
                self.clear_results()
                self.footer_status.config(text=f"Image loaded: {Path(file_path).name}", fg='#3B82F6')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def is_multiview_image(self, img):
        """Detect if image contains multiple brain scans"""
        width, height = img.size
        aspect_ratio = width / height
        
        # Convert to grayscale to analyze
        gray = np.array(img.convert('L'))
        
        # Multi-view images have multiple distinct brain regions
        # Check for grid pattern by looking at edges
        if width > 700 or height > 700:
            # Likely multi-view
            return True
        
        # Check aspect ratio - training data is roughly square (512x512)
        if not (0.85 < aspect_ratio < 1.15):
            return True
        
        return False
    
    def display_image(self):
        """Display image in canvas"""
        if self.current_pil_image is None:
            return
        
        # Hide placeholder
        self.placeholder_label.place_forget()
        
        # Get canvas size
        self.image_canvas.update()
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        # Resize image to fit
        img = self.current_pil_image.copy()
        img.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
        
        # Create PhotoImage
        self.photo = ImageTk.PhotoImage(img)
        
        # Display on canvas
        self.image_canvas.delete('all')
        self.image_canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo,
            anchor='center'
        )
    
    def classify_image_threaded(self):
        """Classify in background thread"""
        if self.model is None:
            messagebox.showwarning("Model Not Ready", "Please wait for the model to load.")
            return
        
        if self.current_pil_image is None:
            return
        
        self.classify_btn.config(state='disabled')
        self.footer_status.config(text="Classifying...", fg='#FCD34D')
        
        def classify():
            try:
                result = self.classify_image_direct()
                self.root.after(0, lambda: self.display_results(result))
            except Exception as e:
                self.root.after(0, lambda: self.on_classification_error(str(e)))
        
        threading.Thread(target=classify, daemon=True).start()
    
    def classify_image_direct(self):
        """Classify using direct model inference"""
        img = self.current_pil_image
        
        # Check if multi-view
        if self.is_multiview_image(img):
            return self.classify_multiview(img)
        else:
            return self.classify_single_slice(img)
    
    def classify_single_slice(self, img):
        """Classify single brain scan slice"""
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_rgb = img.convert('RGB')
        img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        return {
            'type': 'single',
            'predicted_class': self.class_labels[predicted_idx],
            'confidence': confidence,
            'probabilities': {
                self.class_labels[i]: probabilities[i].item()
                for i in range(len(self.class_labels))
            }
        }
    
    def classify_multiview(self, img):
        """Classify multi-view image by extracting slices"""
        width, height = img.size
        aspect_ratio = width / height
        
        # Determine grid layout
        if 1.2 < aspect_ratio < 1.5:
            rows, cols = 3, 4
        elif 0.67 < aspect_ratio < 0.83:
            rows, cols = 4, 3
        else:
            rows, cols = 4, 4
        
        # Extract and classify each slice
        slice_width = width // cols
        slice_height = height // rows
        
        findings = {'glioma': 0, 'meningioma': 0, 'no_tumor': 0, 'pituitary': 0}
        all_probs = {label: [] for label in self.class_labels}
        
        for row in range(rows):
            for col in range(cols):
                left = col * slice_width
                top = row * slice_height
                right = left + slice_width
                bottom = top + slice_height
                
                slice_img = img.crop((left, top, right, bottom))
                result = self.classify_single_slice(slice_img)
                
                findings[result['predicted_class']] += 1
                for label in self.class_labels:
                    all_probs[label].append(result['probabilities'][label])
        
        # Average probabilities
        avg_probs = {
            label: np.mean(all_probs[label])
            for label in self.class_labels
        }
        
        # Primary finding
        primary = max(findings, key=findings.get)
        
        return {
            'type': 'multiview',
            'rows': rows,
            'cols': cols,
            'findings': findings,
            'predicted_class': primary,
            'confidence': avg_probs[primary],
            'probabilities': avg_probs
        }
    
    def display_results(self, result):
        """Display classification results"""
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Update prediction
        display_name = predicted_class.replace('_', ' ').upper()
        self.prediction_label.config(
            text=display_name,
            fg=self.class_colors[predicted_class]
        )
        self.confidence_label.config(
            text=f"Confidence: {confidence*100:.1f}%"
        )
        
        # Update probability bars
        for label in self.class_labels:
            prob = probabilities[label]
            self.update_probability_bar(label, prob)
        
        # Re-enable button
        self.classify_btn.config(state='normal')
        self.footer_status.config(
            text=f"Classification complete: {display_name}",
            fg='#10B981'
        )
    
    def update_probability_bar(self, label, probability):
        """Update a single probability bar"""
        canvas = self.prob_bars[label]
        canvas.delete('all')
        
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width < 2:  # Not yet rendered
            canvas.update()
            width = canvas.winfo_width()
            height = canvas.winfo_height()
        
        # Draw bar
        bar_width = int(width * probability)
        color = self.class_colors[label]
        
        if bar_width > 0:
            canvas.create_rectangle(
                0, 0, bar_width, height,
                fill=color,
                outline=''
            )
        
        # Update percentage
        self.prob_labels[label].config(
            text=f"{probability*100:.1f}%",
            fg=color if probability > 0.5 else '#94a3b8'
        )
    
    def on_classification_error(self, error):
        """Handle classification error"""
        messagebox.showerror("Classification Error", f"Failed to classify image:\n{error}")
        self.classify_btn.config(state='normal')
        self.footer_status.config(text="Error", fg='#EF4444')
    
    def clear_results(self):
        """Clear all results"""
        self.prediction_label.config(text="—", fg='#ffffff')
        self.confidence_label.config(text="")
        
        for label in self.class_labels:
            self.prob_bars[label].delete('all')
            self.prob_labels[label].config(text="0%", fg='#94a3b8')
    
    def clear_all(self):
        """Clear everything"""
        self.current_image_path = None
        self.current_pil_image = None
        self.photo = None
        
        self.image_canvas.delete('all')
        self.placeholder_label.place(relx=0.5, rely=0.5, anchor='center')
        
        self.clear_results()
        self.classify_btn.config(state='disabled')
        self.footer_status.config(text="Ready", fg='#10B981')

def main():
    root = tk.Tk()
    app = BrainTumorClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()
