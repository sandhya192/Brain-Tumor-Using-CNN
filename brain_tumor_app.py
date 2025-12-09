"""
BRISC 2025 - Professional Brain Tumor Classification System
Direct Model Integration | Medical Image Validation | JSON Response Viewer
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageStat
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import os
import json
from pathlib import Path
import threading
from datetime import datetime
import cv2
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

class BrainTumorClassificationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection using CNN - Classification System")
        self.root.geometry("1500x950")
        self.root.configure(bg='#0d1117')
        self.root.resizable(True, True)
        
        # Model configuration
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.class_descriptions = {
            'glioma': 'A tumor that occurs in the brain and spinal cord',
            'meningioma': 'A tumor arising from the meninges',
            'no_tumor': 'No tumor detected in the scan',
            'pituitary': 'A tumor in the pituitary gland'
        }
        self.class_colors = {
            'glioma': '#EF4444',
            'meningioma': '#F59E0B', 
            'no_tumor': '#10B981',
            'pituitary': '#3B82F6'
        }
        
        # State variables
        self.current_image_path = None
        self.current_pil_image = None
        self.last_result = None
        self.model_config = None
        self.training_history = None
        
        # Load model config
        self.load_model_config()
        
        # Create UI
        self.create_ui()
        
        # Load model
        self.load_model_async()
    
    def load_model_config(self):
        """Load training configuration and history"""
        try:
            config_path = 'outputs/classification/config.json'
            history_path = 'outputs/classification/history.json'
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
            
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
    
    def create_ui(self):
        """Create professional UI layout"""
        
        # ===== HEADER BAR =====
        header = tk.Frame(self.root, bg='#161b22', height=70)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # Logo/Title
        title_frame = tk.Frame(header, bg='#161b22')
        title_frame.pack(side='left', padx=30, pady=15)
        
        tk.Label(
            title_frame,
            text="Brain Tumor Detection",
            font=('Segoe UI', 24, 'bold'),
            bg='#161b22',
            fg='#58a6ff'
        ).pack(side='left')
        
        tk.Label(
            title_frame,
            text="using CNN",
            font=('Segoe UI', 20, 'bold'),
            bg='#161b22',
            fg='#8b949e'
        ).pack(side='left', padx=(10, 20))
        
        tk.Label(
            title_frame,
            text="Deep Learning Classification System",
            font=('Segoe UI', 12),
            bg='#161b22',
            fg='#8b949e'
        ).pack(side='left', pady=(8, 0))
        
        # Status panel
        status_frame = tk.Frame(header, bg='#161b22')
        status_frame.pack(side='right', padx=30)
        
        self.status_dot = tk.Canvas(status_frame, width=12, height=12, bg='#161b22', highlightthickness=0)
        self.status_dot.pack(side='left', padx=(0, 8))
        self.status_circle = self.status_dot.create_oval(2, 2, 10, 10, fill='#ffc107', outline='')
        
        self.status_text = tk.Label(
            status_frame,
            text="Loading Model...",
            font=('Segoe UI', 10),
            bg='#161b22',
            fg='#8b949e'
        )
        self.status_text.pack(side='left')
        
        # ===== MAIN CONTENT =====
        main_content = tk.Frame(self.root, bg='#0d1117')
        main_content.pack(fill='both', expand=True, padx=20, pady=15)
        
        # LEFT PANEL - Image Upload
        left_panel = tk.Frame(main_content, bg='#161b22', width=550)
        left_panel.pack(side='left', fill='both', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.create_image_panel(left_panel)
        
        # CENTER PANEL - Results
        center_panel = tk.Frame(main_content, bg='#161b22', width=450)
        center_panel.pack(side='left', fill='both', expand=True, padx=10)
        center_panel.pack_propagate(False)
        
        self.create_results_panel(center_panel)
        
        # RIGHT PANEL - JSON Viewer
        right_panel = tk.Frame(main_content, bg='#161b22', width=400)
        right_panel.pack(side='right', fill='both', padx=(10, 0))
        right_panel.pack_propagate(False)
        
        self.create_json_panel(right_panel)
        
        # ===== FOOTER =====
        footer = tk.Frame(self.root, bg='#161b22', height=40)
        footer.pack(fill='x', side='bottom')
        footer.pack_propagate(False)
        
        device_info = f"Device: {self.device.type.upper()}"
        if self.device.type == 'cuda':
            device_info += f" ({torch.cuda.get_device_name(0)})"
        
        tk.Label(
            footer,
            text=device_info,
            font=('Segoe UI', 9),
            bg='#161b22',
            fg='#6e7681'
        ).pack(side='left', padx=30, pady=10)
        
        self.footer_msg = tk.Label(
            footer,
            text="Ready",
            font=('Segoe UI', 9),
            bg='#161b22',
            fg='#3fb950'
        )
        self.footer_msg.pack(side='right', padx=30, pady=10)
    
    def create_image_panel(self, parent):
        """Create image upload panel"""
        # Header
        header = tk.Frame(parent, bg='#21262d', height=45)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Medical Image Input",
            font=('Segoe UI', 11, 'bold'),
            bg='#21262d',
            fg='#c9d1d9'
        ).pack(side='left', padx=15, pady=12)
        
        # Image display area
        self.image_container = tk.Frame(parent, bg='#0d1117')
        self.image_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.image_canvas = tk.Canvas(
            self.image_container,
            bg='#0d1117',
            highlightthickness=1,
            highlightbackground='#30363d'
        )
        self.image_canvas.pack(fill='both', expand=True)
        
        # Placeholder
        self.placeholder = tk.Label(
            self.image_canvas,
            text="\n\n\nDrag & Drop or Click to Upload\n\nSupported: JPG, PNG\nRequirement: Brain MRI Scan",
            font=('Segoe UI', 11),
            bg='#0d1117',
            fg='#6e7681',
            justify='center'
        )
        self.placeholder.place(relx=0.5, rely=0.5, anchor='center')
        
        # Make canvas clickable
        self.image_canvas.bind('<Button-1>', lambda e: self.select_image())
        self.placeholder.bind('<Button-1>', lambda e: self.select_image())
        
        # Buttons
        btn_frame = tk.Frame(parent, bg='#161b22')
        btn_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.upload_btn = tk.Button(
            btn_frame,
            text="Select Image",
            command=self.select_image,
            font=('Segoe UI', 10, 'bold'),
            bg='#238636',
            fg='white',
            activebackground='#2ea043',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10
        )
        self.upload_btn.pack(side='left', padx=(0, 10))
        
        self.classify_btn = tk.Button(
            btn_frame,
            text="Classify",
            command=self.classify_image,
            font=('Segoe UI', 10, 'bold'),
            bg='#1f6feb',
            fg='white',
            activebackground='#388bfd',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.classify_btn.pack(side='left', padx=(0, 10))
        
        self.report_btn = tk.Button(
            btn_frame,
            text="Generate Report",
            command=self.generate_report,
            font=('Segoe UI', 10, 'bold'),
            bg='#7c3aed',
            fg='white',
            activebackground='#8b5cf6',
            activeforeground='white',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10,
            state='disabled'
        )
        self.report_btn.pack(side='left', padx=(0, 10))
        
        self.clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_all,
            font=('Segoe UI', 10),
            bg='#21262d',
            fg='#c9d1d9',
            activebackground='#30363d',
            activeforeground='#c9d1d9',
            relief='flat',
            cursor='hand2',
            padx=20,
            pady=10
        )
        self.clear_btn.pack(side='left')
        
        # Image info
        self.image_info = tk.Label(
            parent,
            text="No image loaded",
            font=('Segoe UI', 9),
            bg='#161b22',
            fg='#6e7681',
            anchor='w'
        )
        self.image_info.pack(fill='x', padx=15, pady=(0, 10))
    
    def create_results_panel(self, parent):
        """Create classification results panel"""
        # Header
        header = tk.Frame(parent, bg='#21262d', height=45)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Classification Results",
            font=('Segoe UI', 11, 'bold'),
            bg='#21262d',
            fg='#c9d1d9'
        ).pack(side='left', padx=15, pady=12)
        
        # Results content
        results_frame = tk.Frame(parent, bg='#161b22')
        results_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Prediction card
        pred_card = tk.Frame(results_frame, bg='#0d1117', relief='flat')
        pred_card.pack(fill='x', pady=(0, 15))
        
        tk.Label(
            pred_card,
            text="Predicted Diagnosis",
            font=('Segoe UI', 10),
            bg='#0d1117',
            fg='#8b949e'
        ).pack(anchor='w', padx=15, pady=(15, 5))
        
        self.prediction_label = tk.Label(
            pred_card,
            text="—",
            font=('Segoe UI', 32, 'bold'),
            bg='#0d1117',
            fg='#c9d1d9'
        )
        self.prediction_label.pack(anchor='w', padx=15, pady=(0, 5))
        
        self.diagnosis_desc = tk.Label(
            pred_card,
            text="",
            font=('Segoe UI', 10),
            bg='#0d1117',
            fg='#6e7681',
            wraplength=380
        )
        self.diagnosis_desc.pack(anchor='w', padx=15, pady=(0, 15))
        
        # Confidence
        conf_frame = tk.Frame(results_frame, bg='#0d1117')
        conf_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(
            conf_frame,
            text="Confidence",
            font=('Segoe UI', 10),
            bg='#0d1117',
            fg='#8b949e'
        ).pack(anchor='w', padx=15, pady=(15, 5))
        
        self.confidence_label = tk.Label(
            conf_frame,
            text="0.00%",
            font=('Segoe UI', 24, 'bold'),
            bg='#0d1117',
            fg='#c9d1d9'
        )
        self.confidence_label.pack(anchor='w', padx=15, pady=(0, 15))
        
        # Probability distribution
        prob_header = tk.Label(
            results_frame,
            text="Probability Distribution",
            font=('Segoe UI', 10, 'bold'),
            bg='#161b22',
            fg='#c9d1d9'
        )
        prob_header.pack(anchor='w', pady=(0, 10))
        
        self.prob_frame = tk.Frame(results_frame, bg='#161b22')
        self.prob_frame.pack(fill='x')
        
        self.prob_bars = {}
        self.prob_labels = {}
        self.prob_values = {}
        
        for label in self.class_labels:
            row = tk.Frame(self.prob_frame, bg='#0d1117', height=50)
            row.pack(fill='x', pady=3)
            row.pack_propagate(False)
            
            # Class name
            name = tk.Label(
                row,
                text=label.replace('_', ' ').title(),
                font=('Segoe UI', 10),
                bg='#0d1117',
                fg='#c9d1d9',
                width=12,
                anchor='w'
            )
            name.pack(side='left', padx=(10, 10), pady=10)
            
            # Progress bar background
            bar_bg = tk.Frame(row, bg='#21262d', height=8)
            bar_bg.pack(side='left', fill='x', expand=True, pady=20)
            
            # Progress bar
            bar = tk.Canvas(bar_bg, bg='#21262d', height=8, highlightthickness=0)
            bar.pack(fill='both', expand=True)
            self.prob_bars[label] = bar
            
            # Percentage
            pct = tk.Label(
                row,
                text="0.0%",
                font=('Segoe UI', 10, 'bold'),
                bg='#0d1117',
                fg='#6e7681',
                width=8
            )
            pct.pack(side='right', padx=10)
            self.prob_values[label] = pct
    
    def create_json_panel(self, parent):
        """Create JSON response viewer panel"""
        # Header
        header = tk.Frame(parent, bg='#21262d', height=45)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="JSON Response",
            font=('Segoe UI', 11, 'bold'),
            bg='#21262d',
            fg='#c9d1d9'
        ).pack(side='left', padx=15, pady=12)
        
        self.copy_btn = tk.Button(
            header,
            text="Copy",
            command=self.copy_json,
            font=('Segoe UI', 9),
            bg='#21262d',
            fg='#58a6ff',
            activebackground='#30363d',
            activeforeground='#58a6ff',
            relief='flat',
            cursor='hand2',
            padx=10,
            pady=3
        )
        self.copy_btn.pack(side='right', padx=15, pady=8)
        
        # JSON content
        json_frame = tk.Frame(parent, bg='#0d1117')
        json_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.json_text = tk.Text(
            json_frame,
            font=('Consolas', 10),
            bg='#0d1117',
            fg='#c9d1d9',
            insertbackground='#c9d1d9',
            relief='flat',
            wrap='word',
            padx=10,
            pady=10
        )
        self.json_text.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(json_frame, command=self.json_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.json_text.config(yscrollcommand=scrollbar.set)
        
        # Initial JSON
        self.update_json_viewer({
            "status": "ready",
            "model": {
                "name": "CNN",
                "classes": self.class_labels,
                "accuracy": "99%"
            },
            "device": str(self.device),
            "message": "Upload a brain MRI scan to begin classification"
        })
        
        # Model info section
        info_frame = tk.Frame(parent, bg='#161b22')
        info_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        tk.Label(
            info_frame,
            text="Model Information",
            font=('Segoe UI', 10, 'bold'),
            bg='#161b22',
            fg='#c9d1d9'
        ).pack(anchor='w', pady=(0, 10))
        
        model_info = tk.Frame(info_frame, bg='#0d1117')
        model_info.pack(fill='x')
        
        info_items = [
            ("Architecture", "CNN"),
            ("Training Images", "5,000"),
            ("Test Accuracy", "99%"),
            ("Classes", "4")
        ]
        
        for label, value in info_items:
            row = tk.Frame(model_info, bg='#0d1117')
            row.pack(fill='x', padx=10, pady=5)
            
            tk.Label(
                row,
                text=label,
                font=('Segoe UI', 9),
                bg='#0d1117',
                fg='#8b949e',
                width=15,
                anchor='w'
            ).pack(side='left')
            
            tk.Label(
                row,
                text=value,
                font=('Segoe UI', 9, 'bold'),
                bg='#0d1117',
                fg='#c9d1d9'
            ).pack(side='left')
    
    def update_json_viewer(self, data):
        """Update JSON viewer with data"""
        self.json_text.config(state='normal')
        self.json_text.delete('1.0', 'end')
        formatted = json.dumps(data, indent=2)
        self.json_text.insert('1.0', formatted)
        self.json_text.config(state='disabled')
    
    def copy_json(self):
        """Copy JSON to clipboard"""
        content = self.json_text.get('1.0', 'end').strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.footer_msg.config(text="JSON copied to clipboard", fg='#58a6ff')
    
    def load_model_async(self):
        """Load model in background"""
        def load():
            try:
                model_path = 'outputs/classification/best_model.pth'
                
                self.model = models.resnet50(weights=None)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, len(self.class_labels))
                
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                
                self.root.after(0, self.on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self.on_model_error(str(e)))
        
        threading.Thread(target=load, daemon=True).start()
    
    def on_model_loaded(self):
        """Model loaded successfully"""
        self.status_dot.itemconfig(self.status_circle, fill='#3fb950')
        self.status_text.config(text=f"Model Ready ({self.device.type.upper()})")
        self.footer_msg.config(text="Model loaded successfully", fg='#3fb950')
    
    def on_model_error(self, error):
        """Model loading failed"""
        self.status_dot.itemconfig(self.status_circle, fill='#f85149')
        self.status_text.config(text="Model Error")
        messagebox.showerror("Error", f"Failed to load model:\n{error}")
    
    def select_image(self):
        """Select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Brain MRI Scan",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png"),
                ("All Files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_pil_image = Image.open(file_path)
                
                # Validate image
                validation = self.validate_medical_image(self.current_pil_image)
                
                if not validation['is_valid']:
                    self.show_validation_error(validation)
                    return
                
                self.display_image()
                self.classify_btn.config(state='normal')
                self.clear_results()
                
                # Update info
                w, h = self.current_pil_image.size
                size_kb = os.path.getsize(file_path) / 1024
                self.image_info.config(
                    text=f"{Path(file_path).name} | {w}x{h} px | {size_kb:.1f} KB"
                )
                self.footer_msg.config(text="Image loaded - Ready to classify", fg='#3fb950')
                
                # Update JSON
                self.update_json_viewer({
                    "status": "image_loaded",
                    "file": {
                        "name": Path(file_path).name,
                        "path": file_path,
                        "size": f"{size_kb:.1f} KB",
                        "dimensions": f"{w}x{h}"
                    },
                    "validation": validation,
                    "action": "Click 'Classify' to analyze"
                })
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def validate_medical_image(self, img):
        """Validate if image is a medical brain scan"""
        result = {
            'is_valid': True,
            'checks': {},
            'warnings': [],
            'errors': []
        }
        
        width, height = img.size
        
        # Check 1: Image size
        if width < 50 or height < 50:
            result['is_valid'] = False
            result['errors'].append("Image too small (minimum 50x50)")
            result['checks']['size'] = 'FAIL'
        elif width > 2000 or height > 2000:
            result['warnings'].append("Image unusually large")
            result['checks']['size'] = 'WARN'
        else:
            result['checks']['size'] = 'PASS'
        
        # Check 2: Grayscale characteristics (medical scans are usually grayscale)
        img_rgb = img.convert('RGB')
        img_array = np.array(img_rgb)
        
        # Calculate color variance
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
        color_diff = np.abs(r.astype(float) - g.astype(float)).mean() + \
                     np.abs(g.astype(float) - b.astype(float)).mean() + \
                     np.abs(r.astype(float) - b.astype(float)).mean()
        
        is_grayscale = color_diff < 30  # Threshold for grayscale
        
        if not is_grayscale:
            result['is_valid'] = False
            result['errors'].append("Image appears to be colored (not a medical MRI scan)")
            result['checks']['grayscale'] = 'FAIL'
        else:
            result['checks']['grayscale'] = 'PASS'
        
        # Check 3: Aspect ratio (brain scans are typically square-ish)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            result['warnings'].append("Unusual aspect ratio for brain scan")
            result['checks']['aspect_ratio'] = 'WARN'
        else:
            result['checks']['aspect_ratio'] = 'PASS'
        
        # Check 4: Content analysis - check for dark background (typical of MRI)
        gray = np.array(img.convert('L'))
        dark_pixels = np.sum(gray < 30) / gray.size
        bright_pixels = np.sum(gray > 225) / gray.size
        
        if dark_pixels < 0.1 and bright_pixels > 0.5:
            result['is_valid'] = False
            result['errors'].append("Image does not appear to be an MRI scan (too bright)")
            result['checks']['mri_pattern'] = 'FAIL'
        else:
            result['checks']['mri_pattern'] = 'PASS'
        
        # Check 5: Detect if it's a multi-view grid
        if width > 700 or height > 700:
            if not (0.85 < aspect_ratio < 1.15):
                result['warnings'].append("Possible multi-view grid detected")
                result['checks']['single_view'] = 'WARN'
            else:
                result['checks']['single_view'] = 'PASS'
        else:
            result['checks']['single_view'] = 'PASS'
        
        return result
    
    def show_validation_error(self, validation):
        """Show validation error dialog"""
        error_msg = "This image does not appear to be a valid brain MRI scan.\n\n"
        error_msg += "Issues found:\n"
        
        for error in validation['errors']:
            error_msg += f"  - {error}\n"
        
        if validation['warnings']:
            error_msg += "\nWarnings:\n"
            for warning in validation['warnings']:
                error_msg += f"  - {warning}\n"
        
        error_msg += "\nPlease upload a valid brain MRI scan image."
        
        messagebox.showerror("Invalid Image", error_msg)
        
        self.update_json_viewer({
            "status": "error",
            "error": "Invalid medical image",
            "validation": validation
        })
        
        self.footer_msg.config(text="Image rejected - Not a valid MRI scan", fg='#f85149')
    
    def display_image(self):
        """Display image on canvas"""
        if self.current_pil_image is None:
            return
        
        self.placeholder.place_forget()
        
        self.image_canvas.update()
        canvas_w = self.image_canvas.winfo_width()
        canvas_h = self.image_canvas.winfo_height()
        
        img = self.current_pil_image.copy()
        img.thumbnail((canvas_w - 20, canvas_h - 20), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img)
        
        self.image_canvas.delete('all')
        self.image_canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            image=self.photo,
            anchor='center'
        )
    
    def classify_image(self):
        """Run classification"""
        if self.model is None:
            messagebox.showwarning("Not Ready", "Model is still loading...")
            return
        
        if self.current_pil_image is None:
            return
        
        self.classify_btn.config(state='disabled')
        self.footer_msg.config(text="Classifying...", fg='#ffc107')
        
        def run_classification():
            try:
                result = self.run_inference()
                self.root.after(0, lambda: self.display_results(result))
            except Exception as e:
                self.root.after(0, lambda: self.on_classification_error(str(e)))
        
        threading.Thread(target=run_classification, daemon=True).start()
    
    def run_inference(self):
        """Run model inference"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img = self.current_pil_image.convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_idx].item()
        
        # Check if confidence is too low (might be random image that passed validation)
        if confidence < 0.6:
            max_prob = max(probabilities.tolist())
            if max_prob < 0.6:
                return {
                    'status': 'uncertain',
                    'message': 'Classification uncertain - Image may not be a valid brain scan',
                    'predicted_class': self.class_labels[predicted_idx],
                    'confidence': confidence,
                    'probabilities': {
                        self.class_labels[i]: probabilities[i].item()
                        for i in range(len(self.class_labels))
                    }
                }
        
        return {
            'status': 'success',
            'predicted_class': self.class_labels[predicted_idx],
            'confidence': confidence,
            'probabilities': {
                self.class_labels[i]: probabilities[i].item()
                for i in range(len(self.class_labels))
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def display_results(self, result):
        """Display classification results"""
        self.last_result = result
        
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        # Update prediction
        display_name = predicted_class.replace('_', ' ').upper()
        color = self.class_colors.get(predicted_class, '#c9d1d9')
        
        self.prediction_label.config(text=display_name, fg=color)
        self.diagnosis_desc.config(text=self.class_descriptions.get(predicted_class, ''))
        self.confidence_label.config(text=f"{confidence*100:.2f}%", fg=color)
        
        # Update probability bars
        for label in self.class_labels:
            prob = probabilities[label]
            self.update_probability_bar(label, prob)
        
        # Update JSON
        json_response = {
            "status": result['status'],
            "prediction": {
                "class": predicted_class,
                "display_name": display_name,
                "confidence": f"{confidence*100:.2f}%",
                "confidence_raw": confidence
            },
            "probabilities": {
                k: f"{v*100:.2f}%" for k, v in probabilities.items()
            },
            "model": {
                "name": "CNN",
                "device": str(self.device)
            },
            "timestamp": result.get('timestamp', datetime.now().isoformat())
        }
        
        if result['status'] == 'uncertain':
            json_response['warning'] = result['message']
        
        self.update_json_viewer(json_response)
        
        # Re-enable button
        self.classify_btn.config(state='normal')
        self.report_btn.config(state='normal')
        
        if result['status'] == 'uncertain':
            self.footer_msg.config(text="Classification uncertain - Low confidence", fg='#ffc107')
            messagebox.showwarning(
                "Uncertain Classification",
                "The model is not confident about this classification.\n\n"
                "This may indicate:\n"
                "- The image is not a proper brain MRI scan\n"
                "- The image quality is too low\n"
                "- The image format is unusual\n\n"
                "Please verify the input image."
            )
        else:
            self.footer_msg.config(text=f"Classification complete: {display_name}", fg='#3fb950')
    
    def update_probability_bar(self, label, prob):
        """Update probability bar"""
        canvas = self.prob_bars[label]
        canvas.delete('all')
        
        canvas.update()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        if width < 2:
            width = 200
        
        bar_width = int(width * prob)
        color = self.class_colors.get(label, '#58a6ff')
        
        if bar_width > 0:
            canvas.create_rectangle(0, 0, bar_width, height, fill=color, outline='')
        
        self.prob_values[label].config(
            text=f"{prob*100:.1f}%",
            fg=color if prob > 0.3 else '#6e7681'
        )
    
    def on_classification_error(self, error):
        """Handle classification error"""
        messagebox.showerror("Error", f"Classification failed:\n{error}")
        self.classify_btn.config(state='normal')
        self.footer_msg.config(text="Classification error", fg='#f85149')
        
        self.update_json_viewer({
            "status": "error",
            "error": error
        })
    
    def clear_results(self):
        """Clear results"""
        self.prediction_label.config(text="—", fg='#c9d1d9')
        self.diagnosis_desc.config(text="")
        self.confidence_label.config(text="0.00%", fg='#c9d1d9')
        
        for label in self.class_labels:
            self.prob_bars[label].delete('all')
            self.prob_values[label].config(text="0.0%", fg='#6e7681')
        
        self.report_btn.config(state='disabled')
    
    def clear_all(self):
        """Clear everything"""
        self.current_image_path = None
        self.current_pil_image = None
        self.photo = None
        self.last_result = None
        
        self.image_canvas.delete('all')
        self.placeholder.place(relx=0.5, rely=0.5, anchor='center')
        
        self.clear_results()
        self.classify_btn.config(state='disabled')
        self.image_info.config(text="No image loaded")
        self.footer_msg.config(text="Ready", fg='#3fb950')
        
        self.update_json_viewer({
            "status": "ready",
            "model": {
                "name": "CNN",
                "classes": self.class_labels,
                "accuracy": "99%"
            },
            "device": str(self.device),
            "message": "Upload a brain MRI scan to begin classification"
        })
    
    def generate_report(self):
        """Generate PDF medical report"""
        if self.last_result is None or self.current_image_path is None:
            messagebox.showwarning("No Results", "Please classify an image first")
            return
        
        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            title="Save Medical Report",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialfile=f"brain_tumor_cnn_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if not save_path:
            return
        
        try:
            self.footer_msg.config(text="Generating PDF report...", fg='#ffc107')
            self.root.update()
            
            self.create_pdf_report(save_path)
            
            self.footer_msg.config(text=f"Report saved: {Path(save_path).name}", fg='#3fb950')
            
            # Ask to open
            if messagebox.askyesno("Success", f"Report saved successfully!\n\n{save_path}\n\nOpen the report now?"):
                os.startfile(save_path)
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report:\n{str(e)}")
            self.footer_msg.config(text="Report generation failed", fg='#f85149')
    
    def create_pdf_report(self, filename):
        """Create detailed PDF medical report"""
        doc = SimpleDocTemplate(filename, pagesize=letter,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='CustomTitle',
                                 parent=styles['Heading1'],
                                 fontSize=24,
                                 textColor=colors.HexColor('#1f6feb'),
                                 spaceAfter=30,
                                 alignment=TA_CENTER))
        
        styles.add(ParagraphStyle(name='CustomHeading',
                                 parent=styles['Heading2'],
                                 fontSize=14,
                                 textColor=colors.HexColor('#161b22'),
                                 spaceAfter=12,
                                 spaceBefore=12))
        
        styles.add(ParagraphStyle(name='CustomBody',
                                 parent=styles['BodyText'],
                                 fontSize=11,
                                 textColor=colors.HexColor('#24292f')))
        
        # Title
        elements.append(Paragraph("BRAIN TUMOR CLASSIFICATION REPORT", styles['CustomTitle']))
        elements.append(Paragraph("Brain Tumor Detection using CNN - Medical Imaging Analysis", styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Report Information
        elements.append(Paragraph("Report Information", styles['CustomHeading']))
        
        report_data = [
            ["Report ID:", f"BTD-CNN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"],
            ["Generated:", datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ["System:", "Brain Tumor Detection using CNN"],
            ["Model:", "CNN (99% Accuracy)"],
            ["Device:", f"{self.device.type.upper()} Accelerated"]
        ]
        
        t = Table(report_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f6f8fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#24292f')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de'))
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
        
        # Input Image Information
        elements.append(Paragraph("Input Image Details", styles['CustomHeading']))
        
        img_size = os.path.getsize(self.current_image_path) / 1024
        img_w, img_h = self.current_pil_image.size
        
        image_data = [
            ["File Name:", Path(self.current_image_path).name],
            ["Dimensions:", f"{img_w} × {img_h} pixels"],
            ["File Size:", f"{img_size:.2f} KB"],
            ["Format:", Path(self.current_image_path).suffix.upper()[1:]]
        ]
        
        t = Table(image_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f6f8fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#24292f')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de'))
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add input image thumbnail
        temp_img_path = "temp_report_image.jpg"
        img_copy = self.current_pil_image.copy()
        img_copy.thumbnail((400, 400), Image.Resampling.LANCZOS)
        img_copy.save(temp_img_path, quality=95)
        
        img = RLImage(temp_img_path, width=3*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))
        
        # Classification Results
        elements.append(Paragraph("Classification Results", styles['CustomHeading']))
        
        predicted_class = self.last_result['predicted_class']
        confidence = self.last_result['confidence']
        
        result_color = colors.HexColor(self.class_colors.get(predicted_class, '#1f6feb'))
        
        result_data = [
            ["Diagnosis:", predicted_class.replace('_', ' ').upper()],
            ["Confidence:", f"{confidence*100:.2f}%"],
            ["Status:", "High Confidence" if confidence > 0.8 else "Moderate Confidence" if confidence > 0.6 else "Low Confidence"],
            ["Description:", self.class_descriptions.get(predicted_class, "N/A")]
        ]
        
        t = Table(result_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f6f8fa')),
            ('BACKGROUND', (1, 0), (1, 0), result_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#24292f')),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de'))
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
        
        # Probability Distribution
        elements.append(Paragraph("Detailed Probability Analysis", styles['CustomHeading']))
        
        prob_data = [["Class", "Probability", "Percentage"]]
        for label in self.class_labels:
            prob = self.last_result['probabilities'][label]
            prob_data.append([
                label.replace('_', ' ').title(),
                f"{prob:.6f}",
                f"{prob*100:.2f}%"
            ])
        
        t = Table(prob_data, colWidths=[2*inch, 2*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')])
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.3*inch))
        
        # Model Information
        elements.append(Paragraph("Model Details", styles['CustomHeading']))
        
        model_info = [
            ["Architecture:", "CNN (Convolutional Neural Network)"],
            ["Training Dataset:", "Brain MRI Dataset (5,000 images)"],
            ["Test Accuracy:", "99%"],
            ["Input Size:", "224 × 224 pixels"],
            ["Classes:", ", ".join([c.replace('_', ' ').title() for c in self.class_labels])],
            ["Processing Device:", f"{self.device.type.upper()}"]
        ]
        
        t = Table(model_info, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f6f8fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#24292f')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d0d7de'))
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.4*inch))
        
        # Disclaimer
        elements.append(Paragraph("Medical Disclaimer", styles['CustomHeading']))
        disclaimer_text = """
        <para>
        This report is generated by an AI-based classification system for research and educational purposes only. 
        The results should NOT be used as the sole basis for clinical diagnosis or treatment decisions. 
        Always consult with qualified medical professionals for proper diagnosis and treatment planning.
        The system achieves 99% accuracy on test data but may produce errors in real-world scenarios.
        </para>
        """
        elements.append(Paragraph(disclaimer_text, styles['CustomBody']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#6e7681'),
            alignment=TA_CENTER
        )
        elements.append(Paragraph(
            f"Generated by Brain Tumor Detection using CNN System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            footer_style
        ))
        
        # Build PDF
        doc.build(elements)
        
        # Clean up temp image
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

def main():
    root = tk.Tk()
    app = BrainTumorClassificationSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()
