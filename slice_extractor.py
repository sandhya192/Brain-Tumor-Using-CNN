"""
Multi-View Slice Extractor
Extract individual brain scan slices from grid images for accurate classification
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os

class SliceExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain MRI Slice Extractor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a2e')
        
        self.image = None
        self.slices = []
        self.rows = 3
        self.cols = 4
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg='#16213e', height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="Multi-View Slice Extractor",
            font=('Arial', 24, 'bold'),
            bg='#16213e',
            fg='#4F46E5'
        ).pack(pady=25)
        
        # Controls
        control_frame = tk.Frame(self.root, bg='#1a1a2e')
        control_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Button(
            control_frame,
            text="Load Multi-View Image",
            command=self.load_image,
            font=('Arial', 11, 'bold'),
            bg='#4F46E5',
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left', padx=5)
        
        tk.Label(
            control_frame,
            text="Grid Size:",
            font=('Arial', 10),
            bg='#1a1a2e',
            fg='white'
        ).pack(side='left', padx=(20, 5))
        
        tk.Label(
            control_frame,
            text="Rows:",
            font=('Arial', 10),
            bg='#1a1a2e',
            fg='white'
        ).pack(side='left', padx=5)
        
        self.rows_spinbox = tk.Spinbox(
            control_frame,
            from_=1,
            to=6,
            width=5,
            font=('Arial', 10),
            command=self.update_grid
        )
        self.rows_spinbox.delete(0, 'end')
        self.rows_spinbox.insert(0, '3')
        self.rows_spinbox.pack(side='left', padx=5)
        
        tk.Label(
            control_frame,
            text="Cols:",
            font=('Arial', 10),
            bg='#1a1a2e',
            fg='white'
        ).pack(side='left', padx=5)
        
        self.cols_spinbox = tk.Spinbox(
            control_frame,
            from_=1,
            to=6,
            width=5,
            font=('Arial', 10),
            command=self.update_grid
        )
        self.cols_spinbox.delete(0, 'end')
        self.cols_spinbox.insert(0, '4')
        self.cols_spinbox.pack(side='left', padx=5)
        
        tk.Button(
            control_frame,
            text="Extract All Slices",
            command=self.extract_all,
            font=('Arial', 11, 'bold'),
            bg='#10B981',
            fg='white',
            padx=15,
            pady=8
        ).pack(side='left', padx=(20, 5))
        
        # Canvas
        self.canvas = tk.Canvas(
            self.root,
            bg='#0f1419',
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Load a multi-view MRI image to begin",
            font=('Arial', 10),
            bg='#16213e',
            fg='#94a3b8',
            anchor='w',
            padx=20,
            height=2
        )
        self.status_label.pack(fill='x', side='bottom')
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Multi-View MRI Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.image = Image.open(file_path)
                self.image_path = file_path
                self.display_image()
                self.status_label.config(
                    text=f"Loaded: {os.path.basename(file_path)} ({self.image.size[0]}×{self.image.size[1]})"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def update_grid(self):
        self.rows = int(self.rows_spinbox.get())
        self.cols = int(self.cols_spinbox.get())
        self.display_image()
    
    def display_image(self):
        if self.image is None:
            return
        
        self.canvas.delete('all')
        
        # Get canvas size
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Create copy with grid overlay
        img_display = self.image.copy()
        draw = ImageDraw.Draw(img_display)
        
        width, height = img_display.size
        slice_width = width / self.cols
        slice_height = height / self.rows
        
        # Draw grid lines
        for i in range(1, self.cols):
            x = int(i * slice_width)
            draw.line([(x, 0), (x, height)], fill='#4F46E5', width=3)
        
        for i in range(1, self.rows):
            y = int(i * slice_height)
            draw.line([(0, y), (width, y)], fill='#4F46E5', width=3)
        
        # Draw slice numbers
        for row in range(self.rows):
            for col in range(self.cols):
                slice_num = row * self.cols + col + 1
                x = int((col + 0.5) * slice_width)
                y = int((row + 0.5) * slice_height)
                
                # Draw circle background
                r = 25
                draw.ellipse([x-r, y-r, x+r, y+r], fill='#4F46E5', outline='white', width=2)
                
                # Draw number (approximate position)
                draw.text(
                    (x-8, y-12),
                    str(slice_num),
                    fill='white'
                )
        
        # Resize to fit canvas
        img_display.thumbnail((canvas_width - 40, canvas_height - 40), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(img_display)
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo,
            anchor='center'
        )
    
    def extract_all(self):
        if self.image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        try:
            width, height = self.image.size
            slice_width = width // self.cols
            slice_height = height // self.rows
            
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            
            extracted_count = 0
            
            for row in range(self.rows):
                for col in range(self.cols):
                    slice_num = row * self.cols + col + 1
                    
                    left = col * slice_width
                    top = row * slice_height
                    right = left + slice_width
                    bottom = top + slice_height
                    
                    slice_img = self.image.crop((left, top, right, bottom))
                    
                    # Save
                    output_path = os.path.join(
                        output_dir,
                        f"{base_name}_slice_{slice_num:02d}_r{row+1}c{col+1}.jpg"
                    )
                    slice_img.save(output_path, quality=95)
                    extracted_count += 1
            
            self.status_label.config(
                text=f"✓ Extracted {extracted_count} slices to: {output_dir}"
            )
            
            messagebox.showinfo(
                "Success",
                f"Extracted {extracted_count} slices successfully!\n\n"
                f"Output: {output_dir}\n\n"
                "You can now upload individual slices to the classifier."
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract slices:\n{str(e)}")

def main():
    root = tk.Tk()
    app = SliceExtractor(root)
    root.mainloop()

if __name__ == "__main__":
    main()
