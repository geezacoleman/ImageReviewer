import os
import json
import glob
import cv2
import numpy as np
import colorsys
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tempfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

#########################
# Helper Functions
#########################


def load_annotations(annotation_dir):
    annotations_by_filename = {}
    categories = {}
    for json_file in glob.glob(os.path.join(annotation_dir, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)
        image_id_to_filename = {}
        if 'images' in data:
            for image in data['images']:
                image_id_to_filename[image['id']] = image['file_name']
        if 'annotations' in data:
            for ann in data['annotations']:
                img_id = ann.get('image_id')
                filename = image_id_to_filename.get(img_id)
                if filename is None:
                    continue
                annotations_by_filename.setdefault(filename, []).append(ann)
        if 'categories' in data:
            for cat in data['categories']:
                cat_id = cat.get('id')
                cat_name = cat.get('name', str(cat_id))
                if cat_id not in categories:
                    categories[cat_id] = cat_name
    return annotations_by_filename, categories


def generate_category_colors(categories):
    colors = {}
    cat_ids = sorted(categories.keys())
    num_categories = len(cat_ids)
    for i, cat_id in enumerate(cat_ids):
        hue = i / num_categories
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors[cat_id] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


def draw_annotations(image, annotations, categories, category_colors, highlighted_index=None):
    for i, ann in enumerate(annotations):
        cat_id = ann.get('category_id')
        base_color = category_colors.get(cat_id, (0, 255, 0))
        label = categories.get(cat_id, str(cat_id))
        if highlighted_index is not None and i == highlighted_index:
            draw_color = (0, 0, 255)
            thickness = 4
        else:
            draw_color = base_color
            thickness = 2
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation']
            all_polygons = []
            for seg in segmentation:
                if not isinstance(seg, list):
                    continue
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
                all_polygons.append(pts)
                cv2.polylines(image, [pts], isClosed=True, color=draw_color, thickness=thickness)
            if all_polygons:
                combined = np.concatenate(all_polygons, axis=0)
                x, y, w, h = cv2.boundingRect(combined)
                label_position = (x + w // 2, y + h // 2)
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, draw_color, 1, cv2.LINE_AA)
        elif 'bbox' in ann and len(ann['bbox']) == 4:
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(image, (x, y), (x + w, y + h), draw_color, thickness)
            cv2.putText(image, label, (x, max(y - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, draw_color, 1, cv2.LINE_AA)
    return image


def get_settings_file():
    # Save settings in the user's home directory.
    return os.path.join(os.path.expanduser("~"), ".image_review_tool_settings.json")


#########################
# Filtered View Tab with Caching & Debounce
#########################

class FilteredViewTab:
    def __init__(self, parent, app):
        self.parent = parent  # expected to be a Notebook
        self.app = app
        self.filtered_images = []
        self.selected_class = None
        self.instance_images = []  # holds button image references
        self.thumbnail_cache = {}  # cache: (img_file, category_id, ann_index) -> tk image
        self.resize_after_id = None  # for debouncing
        self.loading_task_id = None  # for cancelling loading tasks
        self.is_loading = False  # flag to prevent multiple concurrent loads
        self.max_thumbnails = 100  # limit thumbnails to prevent memory issues
        self.frame = ttk.Frame(self.parent)
        self.parent.add(self.frame, text="Filtered")
        self.create_tab()

    def create_tab(self):
        # Top control panel
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Class selection dropdown
        ttk.Label(control_frame, text="Select Class:", font=("Helvetica", 11)).pack(side=tk.LEFT, padx=5)
        self.class_combobox = ttk.Combobox(control_frame, state="readonly", font=("Helvetica", 11), width=25)
        self.class_combobox.pack(side=tk.LEFT, padx=5)
        self.class_combobox.bind("<<ComboboxSelected>>", self.update_filtered_images)

        # Add a loading indicator and limit control
        limit_frame = ttk.Frame(control_frame)
        limit_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Label(limit_frame, text="Max thumbnails:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.limit_var = tk.StringVar(value=str(self.max_thumbnails))
        limit_entry = ttk.Spinbox(limit_frame, from_=10, to=500, width=5,
                                  textvariable=self.limit_var, increment=10)
        limit_entry.pack(side=tk.LEFT, padx=5)
        self.limit_var.trace("w", self.update_thumbnail_limit)

        # Status/count indicator
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.frame, textvariable=self.status_var, font=("Helvetica", 10))
        status_label.pack(anchor=tk.W, padx=10, pady=(0, 5))

        # Main scrollable area
        self.instance_frame = ttk.Frame(self.frame)
        self.instance_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas for scrolling
        self.canvas = tk.Canvas(self.instance_frame, bg="#f5f5f5", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.instance_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Configure scrolling
        self.scrollable_frame.bind("<Configure>",
                                   lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Grid layout for thumbnails
        for i in range(10):
            self.scrollable_frame.columnconfigure(i, weight=1)

        # Pack canvas and scrollbar
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add progress bar
        self.progress = ttk.Progressbar(self.frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)

        # Add clear cache button
        clear_btn = ttk.Button(self.frame, text="Clear Cache", command=self.clear_thumbnail_cache)
        clear_btn.pack(anchor=tk.E, padx=10, pady=5)

        # Handle canvas resize
        self.canvas.bind("<Configure>", self.on_canvas_resize)

    def clear_thumbnail_cache(self):
        """Clear the thumbnail cache to force regeneration"""
        self.thumbnail_cache.clear()
        if self.selected_class:
            self.update_filtered_images(None)
        messagebox.showinfo("Cache Cleared", "Thumbnail cache has been cleared.")

    def update_thumbnail_limit(self, *args):
        """Update the maximum number of thumbnails to display"""
        try:
            new_limit = int(self.limit_var.get())
            if 10 <= new_limit <= 500:
                self.max_thumbnails = new_limit
                # If already filtered, refresh the view
                if self.selected_class:
                    self.update_filtered_images(None)
        except ValueError:
            pass  # Ignore invalid inputs

    def on_canvas_resize(self, event):
        """Debounce canvas resize to avoid excessive redrawing"""
        if self.resize_after_id:
            self.canvas.after_cancel(self.resize_after_id)
        self.resize_after_id = self.canvas.after(300, self.delayed_resize)

    def delayed_resize(self):
        """Handle resize after debounce period"""
        if self.selected_class and not self.is_loading:
            class_id = self.app.class_name_to_id.get(self.selected_class)
            self.display_cropped_instances(class_id)

    def update_filtered_images(self, event):
        """Filter images by selected class"""
        # Cancel any ongoing loading
        if self.loading_task_id:
            self.canvas.after_cancel(self.loading_task_id)
            self.loading_task_id = None

        # Clear existing thumbnails
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.instance_images = []

        # Get selected class
        self.selected_class = self.class_combobox.get()
        if not self.selected_class:
            self.status_var.set("No class selected")
            return

        # Get class ID and filter images
        class_id = self.app.class_name_to_id.get(self.selected_class)
        self.filtered_images = [img for img in self.app.image_files if any(
            ann["category_id"] == class_id for ann in self.app.annotations_by_filename.get(img, [])
        )]

        # Update status
        if not self.filtered_images:
            self.status_var.set(f"No images contain annotations for '{self.selected_class}'")
            return

        # Set progress bar
        self.progress['value'] = 0
        self.progress['maximum'] = min(len(self.filtered_images), self.max_thumbnails)

        # Start loading thumbnails
        self.is_loading = True
        self.status_var.set(
            f"Loading {min(len(self.filtered_images), self.max_thumbnails)} thumbnails of {len(self.filtered_images)} images...")
        self.display_cropped_instances(class_id)

    def display_cropped_instances(self, class_id):
        """Display cropped instances with limited batch loading for performance"""
        if not self.is_loading:
            # Initial setup for loading
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            self.instance_images = []
            self.canvas.update_idletasks()
            self.is_loading = True
            self.progress['value'] = 0

        # Calculate grid layout
        max_columns = max(1, min(10, self.canvas.winfo_width() // 110))

        # Begin or continue loading
        self.load_batch(class_id, 0, 0, max_columns, 0)

    def load_batch(self, class_id, row, col, max_columns, loaded_count):
        """Load a batch of thumbnails with controlled timing"""
        # Limit total thumbnails
        if loaded_count >= self.max_thumbnails or loaded_count >= len(self.filtered_images):
            self.is_loading = False
            self.status_var.set(f"Showing {loaded_count} thumbnails from {len(self.filtered_images)} images")
            self.progress['value'] = self.progress['maximum']
            return

        # Get the current image file
        img_file = self.filtered_images[loaded_count]
        img_path = os.path.join(self.app.image_dir, img_file)

        try:
            # Get annotations for this class in this image
            matching_annotations = []
            for ann_index, ann in enumerate(self.app.annotations_by_filename.get(img_file, [])):
                if ann.get("category_id") == class_id:
                    matching_annotations.append((ann_index, ann))

            if matching_annotations:
                # Only load the image once for all annotations
                image = cv2.imread(img_path)
                if image is None:
                    # Skip if image can't be loaded
                    self.loading_task_id = self.canvas.after(10,
                                                             lambda: self.load_batch(class_id, row, col, max_columns,
                                                                                     loaded_count + 1))
                    return

                # Process up to 5 annotations per image to avoid overloading
                for ann_index, ann in matching_annotations[:5]:
                    # Create a unique cache key including class_id to avoid mixed thumbnails
                    cache_key = (img_file, class_id, ann_index)

                    # Use cached thumbnail if available
                    if cache_key in self.thumbnail_cache:
                        tk_img = self.thumbnail_cache[cache_key]
                    else:
                        # Create thumbnail from the annotation bounding box
                        if "bbox" in ann and len(ann["bbox"]) == 4:
                            x, y, w, h = map(int, ann["bbox"])

                            # Make sure the bbox coordinates are valid
                            img_h, img_w = image.shape[:2]
                            x = max(0, min(x, img_w - 1))
                            y = max(0, min(y, img_h - 1))
                            w = max(1, min(w, img_w - x))
                            h = max(1, min(h, img_h - y))

                            # Add small margin for visibility
                            x_margin = max(0, x - 5)
                            y_margin = max(0, y - 5)
                            w_margin = min(w + 10, img_w - x_margin)
                            h_margin = min(h + 10, img_h - y_margin)

                            # Extract the cropped region
                            try:
                                cropped = image[y_margin:y_margin + h_margin, x_margin:x_margin + w_margin].copy()

                                # Skip invalid crops
                                if cropped.size == 0 or cropped is None:
                                    continue

                                # Draw a rectangle on the crop to highlight the actual annotation
                                rect_x = x - x_margin
                                rect_y = y - y_margin
                                cv2.rectangle(cropped, (rect_x, rect_y),
                                              (rect_x + w, rect_y + h), (0, 255, 0), 1)

                                # Convert and resize
                                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(cropped)

                                # Calculate aspect ratio for better thumbnail
                                aspect = w_margin / h_margin if h_margin > 0 else 1
                                if aspect > 1.5:  # Wide image
                                    thumb_w, thumb_h = 100, int(100 / aspect)
                                elif aspect < 0.67:  # Tall image
                                    thumb_w, thumb_h = int(100 * aspect), 100
                                else:  # Roughly square
                                    thumb_w, thumb_h = 100, 100

                                # Resize the image
                                try:
                                    pil_img = pil_img.resize((thumb_w, thumb_h), Image.LANCZOS)
                                except AttributeError:
                                    # Fall back for older PIL versions
                                    pil_img = pil_img.resize((thumb_w, thumb_h), Image.ANTIALIAS)

                                # Create a new square image with white background (for consistent thumbnails)
                                square_img = Image.new('RGB', (100, 100), (240, 240, 240))
                                paste_x = (100 - thumb_w) // 2
                                paste_y = (100 - thumb_h) // 2
                                square_img.paste(pil_img, (paste_x, paste_y))

                                # Convert to Tkinter PhotoImage
                                tk_img = ImageTk.PhotoImage(square_img)
                                self.thumbnail_cache[cache_key] = tk_img
                            except Exception as e:
                                print(f"Error creating thumbnail for {img_file}, annotation {ann_index}: {e}")
                                continue
                        else:
                            continue  # Skip annotations without bbox

                    # Create frame to hold thumbnail and label
                    thumb_frame = ttk.Frame(self.scrollable_frame)
                    thumb_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

                    # Create thumbnail button
                    btn = ttk.Button(thumb_frame, image=tk_img,
                                     command=lambda f=img_file, a=ann: self.navigate_to_instance(f, a))
                    btn.pack(pady=(0, 2))

                    # Add small label with filename
                    ttk.Label(thumb_frame, text=img_file[:12] + "..." if len(img_file) > 15 else img_file,
                              font=("Helvetica", 8)).pack()

                    # Store reference to prevent garbage collection
                    self.instance_images.append(tk_img)

                    # Update grid position
                    col += 1
                    if col >= max_columns:
                        col = 0
                        row += 1
        except Exception as e:
            # Log error but continue with other images
            print(f"Error processing {img_file}: {str(e)}")

        # Move to next image
        loaded_count += 1

        # Update progress
        self.progress['value'] = loaded_count

        # Schedule next batch with a slight delay to keep UI responsive
        self.loading_task_id = self.canvas.after(10,
                                                 lambda: self.load_batch(class_id, row, col, max_columns, loaded_count))

    def navigate_to_instance(self, img_file, annotation):
        """Navigate to the selected instance, explicitly saving comments first"""
        # First explicitly save the current comment
        self.app.save_comment()

        # Then navigate to the new image
        self.app.current_index = self.app.image_files.index(img_file)
        self.app.highlighted_annotation_index = self.app.annotations_by_filename[img_file].index(annotation)
        self.app.load_new_image()

    def populate_class_list(self, categories):
        """Populate the class dropdown"""
        if categories:
            self.class_combobox["values"] = list(categories.values())
            self.class_combobox.current(0)
            self.status_var.set(f"Select a class to view {len(self.app.image_files)} images")
        else:
            self.status_var.set("No categories available")


#########################
# Side Panel (Notebook on Right)
#########################

class SidePanel:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # Create a frame to contain everything
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Create the notebook with tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create all tabs
        self.create_load_data_tab()
        self.filtered_tab = FilteredViewTab(self.notebook, self.app)
        self.create_comments_tab()
        self.create_stats_tab()
        self.create_heatmap_tab()

        # Set initial tab
        self.notebook.select(0)  # Start with the first tab

    def create_load_data_tab(self):
        self.load_data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.load_data_frame, text="Load Data")

        # Create a container frame with padding
        container = ttk.Frame(self.load_data_frame, padding="10")
        container.pack(fill=tk.BOTH, expand=True)

        # Grid configuration for better spacing
        for i in range(4):
            container.grid_columnconfigure(i, weight=1)

        # Annotations directory row
        ttk.Label(container, text="Annotations Directory:", font=("Helvetica", 11)).grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=8)
        self.ann_dir_entry = ttk.Entry(container, width=35, font=("Helvetica", 11))
        self.ann_dir_entry.grid(row=0, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        browse_btn1 = ttk.Button(container, text="Browse", command=self.app.browse_ann_dir)
        browse_btn1.grid(row=0, column=3, padx=5, pady=8)

        # Images directory row
        ttk.Label(container, text="Images Directory:", font=("Helvetica", 11)).grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=8)
        self.img_dir_entry = ttk.Entry(container, width=35, font=("Helvetica", 11))
        self.img_dir_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        browse_btn2 = ttk.Button(container, text="Browse", command=self.app.browse_img_dir)
        browse_btn2.grid(row=1, column=3, padx=5, pady=8)

        # Output Excel file row
        ttk.Label(container, text="Output Excel File:", font=("Helvetica", 11)).grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=8)
        self.output_entry = ttk.Entry(container, width=35, font=("Helvetica", 11))
        self.output_entry.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=8)
        browse_btn3 = ttk.Button(container, text="Browse", command=self.app.browse_output_file)
        browse_btn3.grid(row=2, column=3, padx=5, pady=8)

        # Buttons row
        btn_frame = ttk.Frame(container)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=15)

        btn_load = ttk.Button(btn_frame, text="Load Data", command=self.app.load_data, width=15)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_save = ttk.Button(btn_frame, text="Save Settings", command=self.app.save_settings, width=15)
        btn_save.pack(side=tk.LEFT, padx=5)

    # Replace the create_comments_tab method in your SidePanel class

    def create_comments_tab(self):
        self.comments_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comments_frame, text="Comments")

        # Create a vertical paned window for resizable comments area
        comments_paned = tk.PanedWindow(self.comments_frame, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=4)
        comments_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Upper section: Comments editing area
        comments_upper = ttk.Frame(comments_paned)
        comments_paned.add(comments_upper, height=200, stretch="first")

        # Header for comments section
        header_frame = ttk.Frame(comments_upper)
        header_frame.pack(fill=tk.X, anchor=tk.W, padx=5, pady=5)

        ttk.Label(header_frame, text="Comments for current image:",
                  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)

        # Add a save button
        save_btn = ttk.Button(header_frame, text="Save",
                              command=self.app.save_comment, width=8)
        save_btn.pack(side=tk.RIGHT, padx=5)

        # Frame for text area with border
        text_frame = ttk.Frame(comments_upper, borderwidth=1, relief="solid")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.comment_text = tk.Text(text_frame, height=10, font=("Helvetica", 11))
        self.comment_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.comment_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.comment_text.config(yscrollcommand=scrollbar.set)

        # Bind focus out to save comment
        self.comment_text.bind("<FocusOut>", lambda e: self.app.save_comment())

        # Handle Tab key properly
        self.comment_text.bind("<Tab>", self.app.handle_tab_key)

        # Lower section: Comments list/history or additional controls
        comments_lower = ttk.Frame(comments_paned)
        comments_paned.add(comments_lower, height=100)

        # Add a title for the lower section
        ttk.Label(comments_lower, text="Recent Comments:",
                  font=("Helvetica", 11, "bold")).pack(anchor=tk.W, padx=5, pady=5)

        # Add a listbox to show recently commented images
        recent_frame = ttk.Frame(comments_lower, borderwidth=1, relief="solid")
        recent_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.recent_comments = tk.Listbox(recent_frame, font=("Helvetica", 10))
        self.recent_comments.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        recent_scroll = ttk.Scrollbar(recent_frame, orient=tk.VERTICAL, command=self.recent_comments.yview)
        recent_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.recent_comments.config(yscrollcommand=recent_scroll.set)

        # Bind double-click to navigate to the selected image
        self.recent_comments.bind("<Double-1>", self.on_recent_comment_select)

        # Export button with padding at the bottom
        btn_frame = ttk.Frame(self.comments_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        btn_export = ttk.Button(btn_frame, text="Export Reviews", command=self.app.export_reviews, width=15)
        btn_export.pack(side=tk.RIGHT, padx=5)

        # Store references in the app
        self.app.side_panel_comment_text = self.comment_text
        self.app.recent_comments = self.recent_comments

    def on_recent_comment_select(self, event):
        """Navigate to an image from the recent comments list"""
        selection = self.recent_comments.curselection()
        if not selection:
            return

        # Get the selected filename
        selected_item = self.recent_comments.get(selection[0])
        if ":" not in selected_item:
            return

        filename = selected_item.split(":", 1)[0].strip()

        # Save current comment
        self.app.save_comment()

        # Navigate to the selected image
        if filename in self.app.image_files:
            self.app.current_index = self.app.image_files.index(filename)
            self.app.load_new_image()

    def create_stats_tab(self):
        self.stats_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.stats_frame, text="Stats")

        # Add a button to generate statistics
        btn_stats = ttk.Button(self.stats_frame, text="Generate Statistics", command=self.app.generate_stats, width=20)
        btn_stats.pack(pady=10)

        # Frame for summary information
        summary_frame = ttk.LabelFrame(self.stats_frame, text="Summary", padding="5")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)

        self.stats_summary_label = ttk.Label(summary_frame, text="No statistics generated yet.",
                                             font=("Helvetica", 11), justify=tk.LEFT)
        self.stats_summary_label.pack(anchor=tk.W, padx=5, pady=5)

        # Frame for the chart
        self.chart_frame = ttk.Frame(self.stats_frame, borderwidth=1, relief="solid")
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Make these accessible to the main app
        self.app.stats_summary_label = self.stats_summary_label
        self.app.chart_frame = self.chart_frame

    def create_heatmap_tab(self):
        self.heatmap_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.heatmap_frame, text="Heat Map")

        # Controls for generating the heatmap
        controls_frame = ttk.Frame(self.heatmap_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(controls_frame, text="Select Class:", font=("Helvetica", 11)).pack(side=tk.LEFT, padx=5)
        self.heatmap_class_combobox = ttk.Combobox(controls_frame, state="readonly", font=("Helvetica", 11), width=20)
        self.heatmap_class_combobox.pack(side=tk.LEFT, padx=5)

        btn = ttk.Button(controls_frame, text="Generate Heat Map", command=self.app.generate_heatmap, width=15)
        btn.pack(side=tk.LEFT, padx=15)

        # Frame for the heatmap chart
        self.heatmap_chart_frame = ttk.Frame(self.heatmap_frame, borderwidth=1, relief="solid")
        self.heatmap_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)

        # Add initial guidance text
        guidance = ttk.Label(self.heatmap_chart_frame, text="Select a class and click 'Generate Heat Map'",
                             font=("Helvetica", 11), foreground="#555555")
        guidance.pack(expand=True, pady=20)

        # Make these accessible to the main app
        self.app.heatmap_class_combobox = self.heatmap_class_combobox
        self.app.heatmap_chart_frame = self.heatmap_chart_frame

#########################
# Main Application with Three Panes
#########################

class ImageReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Review Tool")

        # Configure ttk style for a more modern look
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TCheckbutton", font=("Helvetica", 10))
        style.configure("Toggle.TButton", font=("Helvetica", 9))

        # Initialize variables
        self.annotation_dir = ""
        self.image_dir = ""
        self.output_excel = ""
        self.annotations_by_filename = {}
        self.categories = {}
        self.category_colors = {}
        self.image_files = []
        self.current_index = 0
        self.comments = {}
        self.annotations_on = tk.BooleanVar(value=True)
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.base_cv_image = None
        self.highlighted_annotation_index = None
        self.class_name_to_id = {}

        self.canvas_width = 800
        self.canvas_height = 600
        self.side_panel_visible = True
        self.filter_var = None  # Will be initialized in create_annotation_panel
        self.counter_var = tk.StringVar(value="Image 0 of 0")  # For image counter
        self.ann_count_var = None  # Will be initialized in create_annotation_panel
        self.photo_image = None  # Will hold the current displayed image
        self.drag_start_x = 0  # For dragging
        self.drag_start_y = 0  # For dragging
        self.orig_pan_x = 0  # For dragging
        self.orig_pan_y = 0  # For dragging
        self.current_image_var = tk.StringVar(value="No image loaded")  # Current image indicator
        self.char_count_var = tk.StringVar(value="0 characters")  # Comment character counter

        # Initialize UI component references that will be set by SidePanel
        self.stats_summary_label = None
        self.chart_frame = None
        self.heatmap_class_combobox = None
        self.heatmap_chart_frame = None
        self.comment_text = None  # For backward compatibility
        self.persistent_comment_text = None  # For the persistent comments pane
        self.side_panel_comment_text = None  # For the side panel comments
        self.recent_comments = None  # For the list of recent comments

        # Create a vertical PanedWindow for the main content + persistent comments
        self.main_vertical_paned = tk.PanedWindow(
            self.root,
            orient=tk.VERTICAL,
            sashrelief=tk.RAISED,
            sashwidth=4,
            sashpad=1
        )
        self.main_vertical_paned.pack(fill=tk.BOTH, expand=True)

        # Create the horizontal PanedWindow for left/center/right panels
        self.main_paned = tk.PanedWindow(
            self.main_vertical_paned,
            orient=tk.HORIZONTAL,
            sashrelief=tk.RAISED,
            sashwidth=4,
            sashpad=1
        )

        # Create the panels
        self.left_panel = ttk.Frame(self.main_paned, width=300)
        self.center_panel = ttk.Frame(self.main_paned)
        self.right_panel = ttk.Frame(self.main_paned, width=300)

        # Add the panels to the horizontal paned window
        self.main_paned.add(self.left_panel, minsize=200, width=300)
        self.main_paned.add(self.center_panel, minsize=400, stretch="always")
        self.main_paned.add(self.right_panel, minsize=250, width=300)

        # Add the horizontal paned window to the vertical paned window
        self.main_vertical_paned.add(self.main_paned, stretch="always")

        # Create the persistent comments pane at the bottom
        self.persistent_comments_pane = ttk.Frame(self.main_vertical_paned)
        self.main_vertical_paned.add(self.persistent_comments_pane, minsize=100, height=150)

        # Comments pane visibility flag
        self.comments_pane_visible = True

        # Initialize all panels
        self.create_annotation_panel(self.left_panel)
        self.create_main_image_area(self.center_panel)
        self.create_navigation_frame(self.center_panel)
        self.create_persistent_comments(self.persistent_comments_pane)

        # Add toggle button at the bottom right of center panel
        self.toggle_side_btn = ttk.Button(
            self.center_panel,
            text="Hide Side Panel ≫",
            command=self.toggle_side_panel,
            style="Toggle.TButton"
        )
        self.toggle_side_btn.pack(side=tk.BOTTOM, anchor=tk.SE, padx=5, pady=5)

        # Toggle button for comments pane
        self.toggle_comments_btn = ttk.Button(
            self.center_panel,
            text="Hide Comments ▲",
            command=self.toggle_comments_pane,
            style="Toggle.TButton"
        )
        self.toggle_comments_btn.pack(side=tk.BOTTOM, anchor=tk.SW, padx=5, pady=5)

        # Initialize the side panel
        self.side_panel = SidePanel(self.right_panel, self)

        # Load settings on startup
        self.load_settings()

        # Set up the close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<Key>", self.on_key)

    def toggle_side_panel(self):
        """Toggle the visibility of the right side panel"""
        if self.side_panel_visible:
            # Hide the right panel
            self.main_paned.forget(self.right_panel)
            self.side_panel_visible = False
            self.toggle_side_btn.config(text="Show Side Panel ≪")
            # Expand center panel to fill the space
            self.main_paned.paneconfigure(self.center_panel, stretch="always")
        else:
            # Reinsert the right panel
            self.main_paned.add(self.right_panel, minsize=250, width=300)
            self.side_panel_visible = True
            self.toggle_side_btn.config(text="Hide Side Panel ≫")

    def create_annotation_panel(self, parent):
        """Create the left panel with annotation list"""
        # Main container with padding
        container = ttk.Frame(parent, padding="8")
        container.pack(fill=tk.BOTH, expand=True)

        # Header with annotation count
        header_frame = ttk.Frame(container)
        header_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(header_frame, text="Annotations:", font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        self.ann_count_var = tk.StringVar(value="(0)")
        ttk.Label(header_frame, textvariable=self.ann_count_var, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=5)

        # Search box for filtering annotations
        search_frame = ttk.Frame(container)
        search_frame.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(search_frame, text="Filter:", font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.filter_var = tk.StringVar()
        self.filter_var.trace("w", self.filter_annotations)
        filter_entry = ttk.Entry(search_frame, textvariable=self.filter_var, width=20)
        filter_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Frame for annotation list with border
        list_frame = ttk.Frame(container, borderwidth=1, relief="solid")
        list_frame.pack(fill=tk.BOTH, expand=True)

        # Listbox for annotations with scrollbar
        self.annotation_listbox = tk.Listbox(
            list_frame,
            font=("Helvetica", 10),
            selectbackground="#4a6984",
            selectforeground="white",
            activestyle="none"
        )
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.annotation_listbox.config(yscrollcommand=scrollbar.set)

        # Bind selection event
        self.annotation_listbox.bind("<<ListboxSelect>>", self.on_annotation_select)
        self.annotation_listbox.bind("<Double-1>", self.center_on_annotation)

        # Add hint text at the bottom
        hint_label = ttk.Label(
            container,
            text="Double-click to center on annotation",
            font=("Helvetica", 8),
            foreground="#555555"
        )
        hint_label.pack(anchor=tk.W, pady=(5, 0))

    def create_main_image_area(self, parent):
        """Create the center panel with the image canvas"""
        # Container for the image area with border
        self.image_frame = ttk.Frame(parent, borderwidth=2, relief="groove")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Status bar for image information
        self.status_frame = ttk.Frame(self.image_frame, height=25)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Image info label
        self.image_info = ttk.Label(self.status_frame, text="No image loaded", font=("Helvetica", 9))
        self.image_info.pack(side=tk.LEFT, padx=5)

        # Zoom info on the right
        self.zoom_label = ttk.Label(self.status_frame, text="Zoom: 1.00x", font=("Helvetica", 9))
        self.zoom_label.pack(side=tk.RIGHT, padx=5)

        # Position info in the middle
        self.position_label = ttk.Label(self.status_frame, text="", font=("Helvetica", 9))
        self.position_label.pack(side=tk.RIGHT, padx=5)

        # Canvas for the image
        self.canvas = tk.Canvas(self.image_frame, bg="#f0f0f0", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<Motion>", self.on_mouse_move)

        # Add a loading indicator/guidance text
        self.loading_text = self.canvas.create_text(
            self.canvas_width // 2,
            self.canvas_height // 2,
            text="No images loaded. Use the 'Load Data' tab to begin.",
            font=("Helvetica", 12),
            fill="#555555"
        )

    def create_navigation_frame(self, parent):
        """Create the navigation controls below the image"""
        # Create a frame with visual separation
        self.nav_frame = ttk.Frame(parent, padding="5")
        self.nav_frame.pack(fill=tk.X, padx=10, pady=5)

        # Left side: Navigation controls
        nav_left = ttk.Frame(self.nav_frame)
        nav_left.pack(side=tk.LEFT)

        # Previous/Next buttons with keyboard shortcut hints
        self.prev_button = ttk.Button(
            nav_left,
            text="← Previous (Left Arrow)",
            command=self.prev_image,
            width=20
        )
        self.prev_button.pack(side=tk.LEFT, padx=(0, 5))

        self.next_button = ttk.Button(
            nav_left,
            text="Next (Right Arrow) →",
            command=self.next_image,
            width=20
        )
        self.next_button.pack(side=tk.LEFT)

        # Center: Image counter
        counter_label = ttk.Label(self.nav_frame, textvariable=self.counter_var, font=("Helvetica", 10))
        counter_label.pack(side=tk.LEFT, padx=20)

        # Right side: Annotation toggle
        nav_right = ttk.Frame(self.nav_frame)
        nav_right.pack(side=tk.RIGHT)

        self.toggle_annotations_btn = ttk.Checkbutton(
            nav_right,
            text="Show Annotations (A)",
            variable=self.annotations_on,
            command=self.toggle_annotations
        )
        self.toggle_annotations_btn.pack(side=tk.RIGHT, padx=5)

        # Add zoom buttons
        zoom_frame = ttk.Frame(nav_right)
        zoom_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Button(zoom_frame, text="−", width=3, command=self.zoom_out).pack(side=tk.LEFT)
        ttk.Label(zoom_frame, text="Zoom", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_frame, text="+", width=3, command=self.zoom_in).pack(side=tk.LEFT)

        # Add reset view button
        ttk.Button(nav_right, text="Reset View", command=self.reset_view).pack(side=tk.RIGHT, padx=5)

    def on_mouse_move(self, event):
        """Update status bar with cursor position"""
        if self.base_cv_image is not None:
            # Calculate the position within the image
            img_x = int((event.x - self.pan_x) / self.zoom_factor)
            img_y = int((event.y - self.pan_y) / self.zoom_factor)

            # Get image dimensions
            h, w = self.base_cv_image.shape[:2]

            # Only show coordinates if cursor is inside the image
            if 0 <= img_x < w and 0 <= img_y < h:
                self.position_label.config(text=f"Pos: ({img_x}, {img_y})")
            else:
                self.position_label.config(text="")

    def on_mouse_press(self, event):
        """Start image dragging"""
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.orig_pan_x = self.pan_x
        self.orig_pan_y = self.pan_y
        self.canvas.config(cursor="fleur")  # Change cursor to indicate dragging

    def on_mouse_drag(self, event):
        """Handle image dragging"""
        if self.base_cv_image is None:
            return

        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x = self.orig_pan_x + dx
        self.pan_y = self.orig_pan_y + dy
        self.refresh_image()

    def on_mouse_wheel(self, event):
        """Handle zoom with mouse wheel"""
        if self.base_cv_image is None:
            return

        # Determine zoom direction based on event
        if hasattr(event, 'delta'):
            # Windows/macOS
            factor = 1.1 if event.delta > 0 else 0.9
        elif event.num == 4:
            # Linux scroll up
            factor = 1.1
        elif event.num == 5:
            # Linux scroll down
            factor = 0.9
        else:
            factor = 1.0

        # Calculate new zoom factor
        new_zoom = self.zoom_factor * factor
        new_zoom = max(0.1, min(new_zoom, 10.0))  # Limit zoom range
        factor = new_zoom / self.zoom_factor

        # Adjust pan to zoom around mouse position
        self.pan_x = event.x - (event.x - self.pan_x) * factor
        self.pan_y = event.y - (event.y - self.pan_y) * factor
        self.zoom_factor = new_zoom

        self.update_zoom_label()
        self.refresh_image()

    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.canvas_width = event.width
        self.canvas_height = event.height

        # If no image is loaded, update the position of the loading text
        if self.base_cv_image is None:
            self.canvas.delete("all")
            self.loading_text = self.canvas.create_text(
                self.canvas_width // 2,
                self.canvas_height // 2,
                text="No images loaded. Use the 'Load Data' tab to begin.",
                font=("Helvetica", 12),
                fill="#555555"
            )
        else:
            self.refresh_image()

    def zoom_in(self):
        """Zoom in on the image"""
        if self.base_cv_image is None:
            return

        factor = 1.25
        cx = self.canvas_width / 2
        cy = self.canvas_height / 2
        new_zoom = self.zoom_factor * factor
        new_zoom = min(new_zoom, 10.0)
        factor = new_zoom / self.zoom_factor
        self.pan_x = cx - (cx - self.pan_x) * factor
        self.pan_y = cy - (cy - self.pan_y) * factor
        self.zoom_factor = new_zoom
        self.update_zoom_label()
        self.refresh_image()

    def zoom_out(self):
        """Zoom out from the image"""
        if self.base_cv_image is None:
            return

        factor = 1 / 1.25
        cx = self.canvas_width / 2
        cy = self.canvas_height / 2
        new_zoom = self.zoom_factor * factor
        new_zoom = max(new_zoom, 0.1)
        factor = new_zoom / self.zoom_factor
        self.pan_x = cx - (cx - self.pan_x) * factor
        self.pan_y = cy - (cy - self.pan_y) * factor
        self.zoom_factor = new_zoom
        self.update_zoom_label()
        self.refresh_image()

    def reset_view(self):
        """Reset zoom and pan to fit the image in the canvas"""
        if self.base_cv_image is None:
            return

        # Calculate best zoom to fit
        orig_h, orig_w = self.base_cv_image.shape[:2]
        self.zoom_factor = min(self.canvas_width / orig_w, self.canvas_height / orig_h, 1.0)

        # Center the image
        new_w = int(orig_w * self.zoom_factor)
        new_h = int(orig_h * self.zoom_factor)
        self.pan_x = (self.canvas_width - new_w) // 2
        self.pan_y = (self.canvas_height - new_h) // 2

        # Update the display
        self.update_zoom_label()
        self.refresh_image()

    def update_zoom_label(self):
        """Update the zoom display in the UI"""
        self.zoom_label.config(text=f"Zoom: {self.zoom_factor:.2f}x")

    def update_counter(self):
        """Update the image counter"""
        if self.image_files:
            self.counter_var.set(f"Image {self.current_index + 1} of {len(self.image_files)}")
        else:
            self.counter_var.set("No images loaded")

    def filter_annotations(self, *args):
        """Filter annotations based on search text"""
        search_text = self.filter_var.get().lower()

        # Store current selection
        current_selection = self.annotation_listbox.curselection()
        selected_index = current_selection[0] if current_selection else None

        # Clear and repopulate the listbox
        self.annotation_listbox.delete(0, tk.END)

        if not self.image_files:
            self.annotation_listbox.insert(tk.END, "No images loaded")
            self.ann_count_var.set("(0)")
            return

        current_file = self.image_files[self.current_index] if self.image_files else None
        if current_file in self.annotations_by_filename:
            filtered_count = 0
            for i, ann in enumerate(self.annotations_by_filename[current_file]):
                cat_id = ann.get("category_id")
                cat_label = self.categories.get(cat_id, str(cat_id))

                # Only add if it matches the filter
                if not search_text or search_text in cat_label.lower():
                    self.annotation_listbox.insert(tk.END, f"{i + 1}: {cat_label}")
                    filtered_count += 1

                    # Color-code the listbox items
                    if cat_id in self.category_colors:
                        color_bgr = self.category_colors.get(cat_id, (0, 255, 0))
                        # Convert BGR to hex for Tkinter
                        color = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])
                        self.annotation_listbox.itemconfig(
                            filtered_count - 1,
                            foreground=color,
                            selectforeground="white"
                        )

            # Update the count label
            total = len(self.annotations_by_filename[current_file])
            if filtered_count < total:
                self.ann_count_var.set(f"({filtered_count}/{total})")
            else:
                self.ann_count_var.set(f"({total})")
        else:
            self.annotation_listbox.insert(tk.END, "No annotations")
            self.ann_count_var.set("(0)")

        # Restore selection if possible
        if selected_index is not None:
            try:
                if selected_index < self.annotation_listbox.size():
                    self.annotation_listbox.selection_set(selected_index)
            except:
                pass

    def update_annotation_list(self):
        """Update the annotation list for the current image"""
        if self.filter_var is not None:  # Make sure filter_var is initialized
            self.filter_annotations()
        else:
            self.annotation_listbox.delete(0, tk.END)
            current_file = self.image_files[self.current_index] if self.image_files else None
            if current_file in self.annotations_by_filename:
                for i, ann in enumerate(self.annotations_by_filename[current_file]):
                    cat_id = ann.get("category_id")
                    cat_label = self.categories.get(cat_id, str(cat_id))
                    self.annotation_listbox.insert(tk.END, f"{i + 1}: {cat_label}")
                self.ann_count_var.set(f"({len(self.annotations_by_filename[current_file])})")
            else:
                self.annotation_listbox.insert(tk.END, "No annotations")
                self.ann_count_var.set("(0)")

    def center_on_annotation(self, event):
        """Center the view on the selected annotation"""
        selection = self.annotation_listbox.curselection()
        if not selection or self.base_cv_image is None:
            return

        self.highlighted_annotation_index = selection[0]
        current_file = self.image_files[self.current_index]

        if current_file in self.annotations_by_filename:
            annotations = self.annotations_by_filename[current_file]
            if self.highlighted_annotation_index < len(annotations):
                ann = annotations[self.highlighted_annotation_index]

                # Get the bounding box
                if "bbox" in ann:
                    x, y, w, h = map(int, ann["bbox"])
                elif "segmentation" in ann and ann["segmentation"]:
                    try:
                        pts = np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2)
                        x, y, w, h = cv2.boundingRect(pts)
                    except:
                        # Refresh and return if cannot get bounds
                        self.refresh_image()
                        return
                else:
                    # Refresh and return if no bounds
                    self.refresh_image()
                    return

                # Center on the annotation
                img_h, img_w = self.base_cv_image.shape[:2]

                # Calculate center point of annotation in image coordinates
                center_x = x + w / 2
                center_y = y + h / 2

                # Calculate where this should be on the canvas
                canvas_center_x = self.canvas_width / 2
                canvas_center_y = self.canvas_height / 2

                # Adjust pan to center the annotation
                self.pan_x = canvas_center_x - (center_x * self.zoom_factor)
                self.pan_y = canvas_center_y - (center_y * self.zoom_factor)

                # Refresh the display
                self.refresh_image()

    def on_annotation_select(self, event):
        """Handle selection of an annotation in the list"""
        selection = self.annotation_listbox.curselection()
        if selection:
            self.highlighted_annotation_index = selection[0]
        else:
            self.highlighted_annotation_index = None
        self.refresh_image()

    def create_persistent_comments(self, parent):
        """Create a persistent comments pane visible at all times"""
        # Container frame with title and controls
        comments_header = ttk.Frame(parent)
        comments_header.pack(fill=tk.X, padx=10, pady=(5, 0))

        # Left side: title
        ttk.Label(comments_header, text="Comments for Current Image:",
                  font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)

        # Right side: image name indicator
        self.current_image_var = tk.StringVar(value="No image loaded")
        ttk.Label(comments_header, textvariable=self.current_image_var,
                  font=("Helvetica", 10, "italic")).pack(side=tk.RIGHT, padx=5)

        # Text area with frame border
        text_frame = ttk.Frame(parent, borderwidth=1, relief="solid")
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        # Text widget for comments with scrollbar - CRITICAL: Name it correctly
        self.persistent_comment_text = tk.Text(text_frame, height=5, font=("Helvetica", 11))
        self.persistent_comment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2, pady=2)

        # For backward compatibility - this ensures both references point to the same widget
        self.comment_text = self.persistent_comment_text

        # Add a vertical scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.persistent_comment_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.persistent_comment_text.config(yscrollcommand=scrollbar.set)

        # Add a character counter
        self.char_count_var = tk.StringVar(value="0 characters")
        char_count = ttk.Label(parent, textvariable=self.char_count_var,
                               font=("Helvetica", 8), foreground="#555555")
        char_count.pack(side=tk.RIGHT, padx=10, pady=(0, 5))

        # Add a "Save" button for explicit saving
        save_btn = ttk.Button(parent, text="Save Comment", command=self.save_comment)
        save_btn.pack(side=tk.LEFT, padx=10, pady=(0, 5))

        # Add a "Clear" button for explicit clearing
        clear_btn = ttk.Button(parent, text="Clear", command=self.clear_comment)
        clear_btn.pack(side=tk.LEFT, padx=10, pady=(0, 5))

        # Bind text change to update character count
        self.persistent_comment_text.bind("<<Modified>>", self.update_char_count)

        # Bind focus out to save comment
        self.persistent_comment_text.bind("<FocusOut>", lambda e: self.save_comment())

        # Handle Tab key properly
        self.persistent_comment_text.bind("<Tab>", self.handle_tab_key)

    def clear_comment(self):
        """Clear the comment in both text areas"""
        if hasattr(self, 'persistent_comment_text') and self.persistent_comment_text:
            self.persistent_comment_text.delete("1.0", tk.END)

        if hasattr(self.side_panel, 'comment_text') and self.side_panel.comment_text:
            self.side_panel.comment_text.delete("1.0", tk.END)

        # If we have a current image, update its comment to empty
        if self.image_files and self.current_index < len(self.image_files):
            current_file = self.image_files[self.current_index]
            self.comments[current_file] = ""

        # Update character count
        if hasattr(self, 'update_char_count'):
            self.update_char_count()

        # Update recent comments list
        if hasattr(self, 'recent_comments') and self.recent_comments:
            self.update_recent_comments_list()

    def update_char_count(self, event=None):
        """Update the character count for the comment text"""
        if hasattr(self, 'persistent_comment_text') and self.persistent_comment_text:
            count = len(self.persistent_comment_text.get("1.0", tk.END).strip())
            self.char_count_var.set(f"{count} characters")
            self.persistent_comment_text.edit_modified(False)  # Reset the modified flag

    def load_new_image(self):
        """Load and display a new image with proper comment clearing"""
        if not self.image_files:
            return

        # Save the current comment before changing image
        self.save_comment()

        # Show loading indicator
        self.canvas.delete("all")
        loading_text = self.canvas.create_text(
            self.canvas_width // 2,
            self.canvas_height // 2,
            text="Loading image...",
            font=("Helvetica", 12),
            fill="#555555"
        )
        self.canvas.update()

        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, current_file)

        try:
            self.base_cv_image = cv2.imread(image_path)
            if self.base_cv_image is None:
                messagebox.showerror("Error", f"Failed to load image: {current_file}")
                return

            # Update image information
            h, w = self.base_cv_image.shape[:2]
            self.image_info.config(text=f"Image: {current_file} ({w}×{h})")

            # Reset highlight
            self.highlighted_annotation_index = None

            # Calculate best zoom to fit
            orig_h, orig_w = self.base_cv_image.shape[:2]
            base_zoom = min(self.canvas_width / orig_w, self.canvas_height / orig_h, 1.0)
            self.zoom_factor = base_zoom
            new_w = int(orig_w * self.zoom_factor)
            new_h = int(orig_h * self.zoom_factor)

            # Center the image
            self.pan_x = (self.canvas_width - new_w) // 2
            self.pan_y = (self.canvas_height - new_h) // 2

            # Update zoom label
            self.update_zoom_label()

            # Update image counter
            self.update_counter()

            # Update annotation panel
            self.update_annotation_list()

            # Refresh the image display
            self.refresh_image()

            # Update current image name in the comments pane
            if hasattr(self, 'current_image_var'):
                self.current_image_var.set(f"Image: {current_file}")

            # CRITICAL: Get the comment for this image (or empty string if none)
            comment_to_load = self.comments.get(current_file, "")

            # CRITICAL: Clear and update persistent comment text
            # This must be done directly with the widget attribute, not through comment_text
            if hasattr(self, 'persistent_comment_text') and self.persistent_comment_text:
                self.persistent_comment_text.delete("1.0", tk.END)  # Clear completely
                if comment_to_load:
                    self.persistent_comment_text.insert(tk.END, comment_to_load)

            # Also update side panel comment text if it exists
            if hasattr(self.side_panel, 'comment_text') and self.side_panel.comment_text:
                self.side_panel.comment_text.delete("1.0", tk.END)  # Clear completely
                if comment_to_load:
                    self.side_panel.comment_text.insert(tk.END, comment_to_load)

            # Update character count
            if hasattr(self, 'update_char_count'):
                self.update_char_count()

            # Update recent comments list if available
            if hasattr(self, 'recent_comments') and self.recent_comments:
                self.update_recent_comments_list()

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image {current_file}: {str(e)}")
            print(f"Error loading image: {str(e)}")  # Additional console debug info

    def save_comment(self):
        """Save the current comment from whichever text area was last used"""
        if not self.image_files:
            return

        current_file = self.image_files[self.current_index]

        # Initialize with empty comment
        comment = ""
        side_panel_comment = ""
        persistent_comment = ""

        # Get comment from the persistent comment pane
        if hasattr(self, 'persistent_comment_text') and self.persistent_comment_text:
            persistent_comment = self.persistent_comment_text.get("1.0", tk.END).strip()
            comment = persistent_comment

        # Get comment from the side panel
        if hasattr(self.side_panel, 'comment_text') and self.side_panel.comment_text:
            side_panel_comment = self.side_panel.comment_text.get("1.0", tk.END).strip()

            # If side panel has newer content and is focused or was last modified
            # (Note: This heuristic might need adjustment based on your exact UI interaction patterns)
            focused_widget = self.root.focus_get()
            if (focused_widget == self.side_panel.comment_text or
                    (persistent_comment == self.comments.get(current_file,
                                                             "") and side_panel_comment != persistent_comment)):
                comment = side_panel_comment

        # Save the comment
        self.comments[current_file] = comment

        # Synchronize both text areas
        if hasattr(self, 'persistent_comment_text') and self.persistent_comment_text:
            if self.persistent_comment_text.get("1.0", tk.END).strip() != comment:
                self.persistent_comment_text.delete("1.0", tk.END)
                self.persistent_comment_text.insert(tk.END, comment)

        if hasattr(self.side_panel, 'comment_text') and self.side_panel.comment_text:
            if self.side_panel.comment_text.get("1.0", tk.END).strip() != comment:
                self.side_panel.comment_text.delete("1.0", tk.END)
                self.side_panel.comment_text.insert(tk.END, comment)

        # Update recent comments list if available
        if hasattr(self, 'recent_comments') and self.recent_comments:
            self.update_recent_comments_list()

    def next_image(self):
        """Navigate to the next image"""
        if not self.image_files:
            return

        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_new_image()

    def prev_image(self):
        """Navigate to the previous image"""
        if not self.image_files:
            return

        if self.current_index > 0:
            self.current_index -= 1
            self.load_new_image()

    def handle_tab_key(self, event):
        """Handle tab key in text widgets to allow focus navigation"""
        event.widget.tk_focusNext().focus()
        return "break"

    def toggle_comments_pane(self):
        """Toggle the visibility of the comments pane"""
        if self.comments_pane_visible:
            # Hide the comments pane
            self.main_vertical_paned.forget(self.persistent_comments_pane)
            self.comments_pane_visible = False
            self.toggle_comments_btn.config(text="Show Comments ▼")
        else:
            # Show the comments pane
            self.main_vertical_paned.add(self.persistent_comments_pane, minsize=100, height=150)
            self.comments_pane_visible = True
            self.toggle_comments_btn.config(text="Hide Comments ▲")

    def refresh_image(self):
        """Update the displayed image with current state"""
        if self.base_cv_image is None:
            return

        # Make a copy of the original image
        img = self.base_cv_image.copy()
        current_file = self.image_files[self.current_index]

        # Draw annotations if enabled
        if self.annotations_on.get() and current_file in self.annotations_by_filename:
            ann_list = self.annotations_by_filename[current_file]
            img = draw_annotations(img, ann_list, self.categories, self.category_colors,
                                   highlighted_index=self.highlighted_annotation_index)

        # Convert from OpenCV BGR to RGB for PIL
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img)

        # Apply zoom
        orig_w, orig_h = pil_image.size
        new_w = int(orig_w * self.zoom_factor)
        new_h = int(orig_h * self.zoom_factor)

        # Use LANCZOS resampling for better quality (replaces deprecated ANTIALIAS)
        try:
            zoomed_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        except AttributeError:
            # Fall back to ANTIALIAS for older PIL versions
            zoomed_image = pil_image.resize((new_w, new_h), Image.ANTIALIAS)

        # Create Tkinter PhotoImage and display
        self.photo_image = ImageTk.PhotoImage(zoomed_image)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo_image)

        # Reset cursor after dragging
        self.canvas.config(cursor="")

    def update_recent_comments_list(self):
        """Update the recent comments list in the comments tab"""
        if not hasattr(self, 'recent_comments') or self.recent_comments is None:
            return

        # Store the current selection if any
        current_selection = None
        if self.recent_comments.curselection():
            current_selection = self.recent_comments.get(self.recent_comments.curselection()[0])

        # Clear the list
        self.recent_comments.delete(0, tk.END)

        # Get commented images (only with non-empty comments)
        commented_files = []
        for img_file, comment in self.comments.items():
            if comment.strip():  # Only include non-empty comments
                commented_files.append((img_file, comment))

        # Sort alphabetically by filename for consistent display
        for img_file, comment in sorted(commented_files, key=lambda x: x[0]):
            # Create a clean preview (single line, truncated)
            preview = comment.replace("\n", " ").strip()
            if len(preview) > 30:
                preview = preview[:30] + "..."

            # Add to the listbox
            self.recent_comments.insert(tk.END, f"{img_file}: {preview}")

        # Restore selection if possible
        if current_selection:
            for i in range(self.recent_comments.size()):
                if self.recent_comments.get(i) == current_selection:
                    self.recent_comments.selection_set(i)
                    self.recent_comments.see(i)
                    break

    def toggle_annotations(self):
        """Toggle annotation display"""
        self.refresh_image()

    def on_key(self, event):
        """Handle keyboard shortcuts"""
        # Check if focus is in a Text widget (comment area)
        focused_widget = self.root.focus_get()
        if isinstance(focused_widget, tk.Text):
            # Let the Text widget handle the key normally
            return

        # Otherwise, handle navigation shortcuts
        if event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()
        elif event.char.lower() == "a":
            self.annotations_on.set(not self.annotations_on.get())
            self.toggle_annotations()
        elif event.keysym in ("plus", "equal"):
            self.zoom_in()
        elif event.keysym in ("minus", "KP_Subtract"):
            self.zoom_out()
        elif event.keysym == "r":
            self.reset_view()

    def export_reviews(self):
        """Export comments to Excel with improved information"""
        self.save_comment()  # Make sure current comment is saved

        if not self.comments:
            messagebox.showinfo("No Comments", "No comments to export.")
            return

        # Gather comment data with additional information
        data = []

        for image_file in self.image_files:
            # Get annotation info
            anns = self.annotations_by_filename.get(image_file, [])
            ann_classes = []
            for ann in anns:
                cat_id = ann.get('category_id')
                cat_label = self.categories.get(cat_id, str(cat_id))
                if cat_label not in ann_classes:
                    ann_classes.append(cat_label)

            # Create entry with comment and annotation info
            data.append({
                "image_name": image_file,
                "comment": self.comments.get(image_file, ""),
                "has_comment": image_file in self.comments and bool(self.comments[image_file].strip()),
                "annotation_count": len(anns),
                "annotation_classes": ", ".join(ann_classes)
            })

        # Convert to DataFrame
        df = pd.DataFrame(data)

        try:
            # Export to Excel with formatting
            with pd.ExcelWriter(self.output_excel, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Image Reviews')

                # Get the workbook and the worksheet
                workbook = writer.book
                worksheet = writer.sheets['Image Reviews']

                # Set column widths
                worksheet.column_dimensions['A'].width = 25  # image_name
                worksheet.column_dimensions['B'].width = 50  # comment
                worksheet.column_dimensions['C'].width = 12  # has_comment
                worksheet.column_dimensions['D'].width = 15  # annotation_count
                worksheet.column_dimensions['E'].width = 30  # annotation_classes

            messagebox.showinfo("Export", f"Reviews exported to {self.output_excel}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def generate_stats(self):
        total_images = len(self.image_files)
        total_annotations = 0
        per_class_counts = {}
        for anns in self.annotations_by_filename.values():
            total_annotations += len(anns)
            for ann in anns:
                cat_id = ann.get('category_id')
                per_class_counts[cat_id] = per_class_counts.get(cat_id, 0) + 1
        avg_annotations = total_annotations / total_images if total_images else 0
        summary_text = (f"Total Images: {total_images}\n"
                        f"Total Annotations: {total_annotations}\n"
                        f"Avg Annotations/Image: {avg_annotations:.2f}")
        self.stats_summary_label.config(text=summary_text)
        class_labels = []
        counts = []
        for cat_id, count in per_class_counts.items():
            label = self.categories.get(cat_id, str(cat_id))
            class_labels.append(label)
            counts.append(count)
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(class_labels, counts, color='skyblue')
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        ax.set_ylabel("Count")
        ax.set_title("Annotations per Class")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout(pad=3)
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.chart_canvas, self.chart_frame)
        toolbar.update()
        self.chart_canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

    def generate_heatmap(self):
        selected_class = self.heatmap_class_combobox.get()
        if not selected_class:
            messagebox.showerror("Error", "Please select a class.")
            return
        class_id = self.class_name_to_id.get(selected_class)
        if class_id is None:
            messagebox.showerror("Error", "Selected class not found.")
            return
        x_centers = []
        y_centers = []
        for image_file in self.image_files:
            if image_file in self.annotations_by_filename:
                image_path = os.path.join(self.image_dir, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                for ann in self.annotations_by_filename[image_file]:
                    if ann.get("category_id") == class_id:
                        if "bbox" in ann and len(ann["bbox"]) == 4:
                            x, y, w, h = ann["bbox"]
                        elif "segmentation" in ann and ann["segmentation"]:
                            try:
                                pts = np.array(ann["segmentation"][0], dtype=np.float32).reshape(-1, 2)
                                x, y, w, h = cv2.boundingRect(pts)
                            except Exception:
                                continue
                        else:
                            continue
                        center_x = (x + w / 2) / img_w
                        center_y = (y + h / 2) / img_h
                        x_centers.append(center_x)
                        y_centers.append(center_y)
        if not x_centers:
            messagebox.showinfo("Info", "No annotations found for selected class.")
            return
        bins = 50
        heatmap, xedges, yedges = np.histogram2d(x_centers, y_centers, bins=bins, range=[[0, 1], [0, 1]])
        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.imshow(heatmap.T, origin='lower', extent=[0, 1, 0, 1], cmap='hot', aspect='auto')
        ax.set_title(f"Heat Map for Class: {selected_class}")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        fig.colorbar(cax, ax=ax)
        fig.tight_layout(pad=3)
        for widget in self.heatmap_chart_frame.winfo_children():
            widget.destroy()
        self.heatmap_chart_canvas = FigureCanvasTkAgg(fig, master=self.heatmap_chart_frame)
        self.heatmap_chart_canvas.draw()
        self.heatmap_chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.heatmap_chart_canvas, self.heatmap_chart_frame)
        toolbar.update()
        self.heatmap_chart_canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)

    def load_settings(self):
        settings_file = get_settings_file()
        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                self.annotation_dir = settings.get("annotation_dir", "")
                self.image_dir = settings.get("image_dir", "")
                self.output_excel = settings.get("output_excel", "")
                try:
                    self.side_panel.ann_dir_entry.delete(0, tk.END)
                    self.side_panel.ann_dir_entry.insert(0, self.annotation_dir)
                    self.side_panel.img_dir_entry.delete(0, tk.END)
                    self.side_panel.img_dir_entry.insert(0, self.image_dir)
                    self.side_panel.output_entry.delete(0, tk.END)
                    self.side_panel.output_entry.insert(0, self.output_excel)
                except Exception:
                    pass
            except Exception as e:
                messagebox.showwarning("Settings Load Error", f"Could not load settings: {e}")

    def save_settings(self):
        try:
            self.annotation_dir = self.side_panel.ann_dir_entry.get().strip()
            self.image_dir = self.side_panel.img_dir_entry.get().strip()
            self.output_excel = self.side_panel.output_entry.get().strip()
        except Exception:
            pass
        settings = {
            "annotation_dir": self.annotation_dir,
            "image_dir": self.image_dir,
            "output_excel": self.output_excel
        }
        settings_file = get_settings_file()
        try:
            with open(settings_file, "w") as f:
                json.dump(settings, f)
            messagebox.showinfo("Settings Saved", "Settings have been saved successfully.")
        except Exception as e:
            messagebox.showerror("Settings Save Error", f"Could not save settings: {e}")

    def on_closing(self):
        self.save_settings()
        self.root.destroy()

    def browse_ann_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            try:
                self.side_panel.ann_dir_entry.delete(0, tk.END)
                self.side_panel.ann_dir_entry.insert(0, directory)
            except Exception:
                pass
            self.annotation_dir = directory

    def browse_img_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            try:
                self.side_panel.img_dir_entry.delete(0, tk.END)
                self.side_panel.img_dir_entry.insert(0, directory)
            except Exception:
                pass
            self.image_dir = directory

    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            try:
                self.side_panel.output_entry.delete(0, tk.END)
                self.side_panel.output_entry.insert(0, file_path)
            except Exception:
                pass
            self.output_excel = file_path

    def load_data(self):
        self.annotation_dir = self.side_panel.ann_dir_entry.get().strip()
        self.image_dir = self.side_panel.img_dir_entry.get().strip()
        self.output_excel = self.side_panel.output_entry.get().strip()
        if not os.path.isdir(self.annotation_dir):
            messagebox.showerror("Error", "Invalid annotations directory.")
            return
        if not os.path.isdir(self.image_dir):
            messagebox.showerror("Error", "Invalid images directory.")
            return
        if not self.output_excel.endswith(".xlsx"):
            messagebox.showerror("Error", "Output file must have a .xlsx extension.")
            return
        self.annotations_by_filename, self.categories = load_annotations(self.annotation_dir)
        self.category_colors = generate_category_colors(self.categories)
        self.side_panel.filtered_tab.populate_class_list(self.categories)
        self.class_name_to_id = {}
        if self.categories:
            class_names = []
            for cat_id, cat_name in self.categories.items():
                class_names.append(cat_name)
                self.class_name_to_id[cat_name] = cat_id
            self.heatmap_class_combobox['values'] = class_names
            if class_names:
                self.heatmap_class_combobox.current(0)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.lower().endswith(valid_exts)])
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the image directory.")
            return
        self.current_index = 0
        self.comments = {}
        self.update_counter()
        self.load_new_image()

#########################
# Main Entry
#########################

if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")
    app = ImageReviewApp(root)
    root.bind("<Key>", app.on_key)
    root.mainloop()
