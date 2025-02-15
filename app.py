import os
import json
import glob
import cv2
import numpy as np
import colorsys
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# For the statistics and heat map charts:
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#########################
# Helper Functions
#########################

def load_annotations(annotation_dir):
    """
    Loads all COCO-format JSON files from the annotation directory.
    Returns:
      - annotations_by_filename: dict mapping image file names to a list of annotation dicts.
      - categories: dict mapping category_id to category name.
    """
    annotations_by_filename = {}
    categories = {}

    for json_file in glob.glob(os.path.join(annotation_dir, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Build mapping from image id to filename.
        image_id_to_filename = {}
        if 'images' in data:
            for image in data['images']:
                image_id_to_filename[image['id']] = image['file_name']

        # Group annotations by filename.
        if 'annotations' in data:
            for ann in data['annotations']:
                img_id = ann.get('image_id')
                filename = image_id_to_filename.get(img_id)
                if filename is None:
                    continue
                annotations_by_filename.setdefault(filename, []).append(ann)

        # Process categories.
        if 'categories' in data:
            for cat in data['categories']:
                cat_id = cat.get('id')
                cat_name = cat.get('name', str(cat_id))
                if cat_id not in categories:
                    categories[cat_id] = cat_name

    return annotations_by_filename, categories


def generate_category_colors(categories):
    """
    Generate a distinct BGR color for each category.
    Returns a dict mapping category_id to a BGR tuple.
    """
    colors = {}
    cat_ids = sorted(categories.keys())
    num_categories = len(cat_ids)
    for i, cat_id in enumerate(cat_ids):
        hue = i / num_categories  # equally spaced hues
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        color = (int(b * 255), int(g * 255), int(r * 255))  # OpenCV uses BGR
        colors[cat_id] = color
    return colors


def draw_annotations(image, annotations, categories, category_colors, highlighted_index=None):
    """
    Draw segmentation annotations (or bounding boxes as a fallback) on the image.
    If 'highlighted_index' is provided, that annotation is drawn with a red, thicker outline.
    """
    for i, ann in enumerate(annotations):
        cat_id = ann.get('category_id')
        base_color = category_colors.get(cat_id, (0, 255, 0))
        label = categories.get(cat_id, str(cat_id))

        if highlighted_index is not None and i == highlighted_index:
            draw_color = (0, 0, 255)  # red for highlight
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
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
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


#########################
# Tkinter Application
#########################

class ImageReviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Review Tool")

        # State variables.
        self.annotation_dir = ""
        self.image_dir = ""
        self.output_excel = ""
        self.annotations_by_filename = {}
        self.categories = {}
        self.category_colors = {}
        self.image_files = []
        self.current_index = 0
        self.comments = {}  # {image_filename: comment}
        self.annotations_on = tk.BooleanVar(value=True)
        self.zoom_factor = 1.0  # current zoom (preserved on toggle)
        self.pan_x = 0
        self.pan_y = 0
        self.base_cv_image = None  # original image (cv2 BGR)
        self.highlighted_annotation_index = None  # index of highlighted annotation

        # For the heat map tab.
        self.class_name_to_id = {}

        # Default canvas size.
        self.canvas_width = 800
        self.canvas_height = 600

        # Build UI.
        self.create_main_image_area()  # Top: main image and annotation panel
        self.create_navigation_frame()  # Below image: navigation buttons
        self.create_bottom_notebook()  # Bottom: Tabs for Load Data, Comments, Statistics, Heat Map

        # Bind keyboard shortcuts.
        self.root.bind("<Key>", self.on_key)

    #########################
    # Main Image Area & Annotation Panel
    #########################

    def create_main_image_area(self):
        """Create the top area containing the image canvas (left) and annotation panel (right)."""
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Image Frame.
        self.image_frame = tk.Frame(self.main_frame, bd=2, relief=tk.SUNKEN)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        # Annotation Panel.
        self.annotation_panel = tk.Frame(self.main_frame)
        self.annotation_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        tk.Label(self.annotation_panel, text="Annotations:", font=("Helvetica", 12, "bold")).pack(anchor=tk.NW)
        self.annotation_listbox = tk.Listbox(self.annotation_panel, width=30, font=("Helvetica", 12))
        self.annotation_listbox.pack(fill=tk.BOTH, expand=True)
        self.annotation_listbox.bind("<<ListboxSelect>>", self.on_annotation_select)

    #########################
    # Navigation Frame
    #########################

    def create_navigation_frame(self):
        """Frame with navigation buttons, annotation toggle, and zoom indicator."""
        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(fill=tk.X, padx=10, pady=5)

        self.prev_button = tk.Button(self.nav_frame, text="Previous", command=self.prev_image,
                                     font=("Helvetica", 14), width=10, height=2)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.nav_frame, text="Next", command=self.next_image,
                                     font=("Helvetica", 14), width=10, height=2)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.toggle_button = tk.Checkbutton(self.nav_frame, text="Show Annotations",
                                            variable=self.annotations_on,
                                            command=self.toggle_annotations,
                                            font=("Helvetica", 14), width=15, height=2)
        self.toggle_button.pack(side=tk.LEFT, padx=20)

        self.zoom_label = tk.Label(self.nav_frame, text=f"Zoom: {self.zoom_factor:.2f}x",
                                   font=("Helvetica", 14))
        self.zoom_label.pack(side=tk.RIGHT, padx=5)

    #########################
    # Bottom Notebook (Tabs)
    #########################

    def create_bottom_notebook(self):
        """Create a Notebook with four tabs: Load Data, Comments, Statistics, Heat Map."""
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        self.notebook = ttk.Notebook(self.bottom_frame, height=250)
        self.notebook.pack(fill=tk.X, expand=True)

        # Load Data Tab.
        self.load_data_tab = tk.Frame(self.notebook)
        self.notebook.add(self.load_data_tab, text="Load Data")
        self.create_load_data_tab_contents(self.load_data_tab)

        # Comments Tab.
        self.comments_tab = tk.Frame(self.notebook)
        self.notebook.add(self.comments_tab, text="Comments")
        self.create_comments_tab_contents(self.comments_tab)

        # Statistics Tab.
        self.stats_tab = tk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        self.create_stats_tab_contents(self.stats_tab)

        # Heat Map Tab.
        self.heatmap_tab = tk.Frame(self.notebook)
        self.notebook.add(self.heatmap_tab, text="Heat Map")
        self.create_heatmap_tab_contents(self.heatmap_tab)

    def create_load_data_tab_contents(self, parent):
        """Populate the Load Data tab with configuration widgets."""
        lbl1 = tk.Label(parent, text="Annotations Directory:", font=("Helvetica", 12))
        lbl1.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ann_dir_entry = tk.Entry(parent, width=40, font=("Helvetica", 12))
        self.ann_dir_entry.grid(row=0, column=1, padx=5, pady=2)
        btn1 = tk.Button(parent, text="Browse", command=self.browse_ann_dir,
                         font=("Helvetica", 12), width=10)
        btn1.grid(row=0, column=2, padx=5, pady=2)

        lbl2 = tk.Label(parent, text="Images Directory:", font=("Helvetica", 12))
        lbl2.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.img_dir_entry = tk.Entry(parent, width=40, font=("Helvetica", 12))
        self.img_dir_entry.grid(row=1, column=1, padx=5, pady=2)
        btn2 = tk.Button(parent, text="Browse", command=self.browse_img_dir,
                         font=("Helvetica", 12), width=10)
        btn2.grid(row=1, column=2, padx=5, pady=2)

        lbl3 = tk.Label(parent, text="Output Excel File:", font=("Helvetica", 12))
        lbl3.grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.output_entry = tk.Entry(parent, width=40, font=("Helvetica", 12))
        self.output_entry.grid(row=2, column=1, padx=5, pady=2)
        btn3 = tk.Button(parent, text="Browse", command=self.browse_output_file,
                         font=("Helvetica", 12), width=10)
        btn3.grid(row=2, column=2, padx=5, pady=2)

        btn_load = tk.Button(parent, text="Load Data", command=self.load_data,
                             font=("Helvetica", 14), width=15)
        btn_load.grid(row=3, column=1, pady=10)

    def create_comments_tab_contents(self, parent):
        """Populate the Comments tab with a text widget and Export Reviews button."""
        lbl = tk.Label(parent, text="Enter Comments for the current image below:",
                       font=("Helvetica", 12, "bold"))
        lbl.pack(anchor=tk.W, padx=5, pady=2)
        self.comment_text = tk.Text(parent, height=6, font=("Helvetica", 12))
        self.comment_text.pack(fill=tk.X, padx=5, pady=2)
        btn_export = tk.Button(parent, text="Export Reviews", command=self.export_reviews,
                               font=("Helvetica", 14), width=15)
        btn_export.pack(pady=5)

    def create_stats_tab_contents(self, parent):
        """Populate the Statistics tab with a smaller Generate Stats button and a large chart area."""
        btn_stats = tk.Button(parent, text="Generate Stats", command=self.generate_stats,
                              font=("Helvetica", 12), width=12, height=1)
        btn_stats.pack(pady=5)
        self.stats_summary_label = tk.Label(parent, text="", font=("Helvetica", 12), justify=tk.LEFT)
        self.stats_summary_label.pack(anchor=tk.NW, padx=5)
        self.chart_frame = tk.Frame(parent)
        self.chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_heatmap_tab_contents(self, parent):
        """Populate the Heat Map tab with a class selection and Generate button and a chart area."""
        controls_frame = tk.Frame(parent)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        lbl = tk.Label(controls_frame, text="Select Class:", font=("Helvetica", 12))
        lbl.pack(side=tk.LEFT, padx=5)
        self.heatmap_class_combobox = ttk.Combobox(controls_frame, state="readonly", font=("Helvetica", 12))
        self.heatmap_class_combobox.pack(side=tk.LEFT, padx=5)
        btn = tk.Button(controls_frame, text="Generate Heat Map", command=self.generate_heatmap,
                        font=("Helvetica", 12), width=15)
        btn.pack(side=tk.LEFT, padx=5)
        self.heatmap_chart_frame = tk.Frame(parent)
        self.heatmap_chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    #########################
    # Callback and Helper Methods
    #########################

    def browse_ann_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.ann_dir_entry.delete(0, tk.END)
            self.ann_dir_entry.insert(0, directory)

    def browse_img_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.img_dir_entry.delete(0, tk.END)
            self.img_dir_entry.insert(0, directory)

    def browse_output_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, file_path)

    def load_data(self):
        """Load annotations and images based on user input."""
        self.annotation_dir = self.ann_dir_entry.get().strip()
        self.image_dir = self.img_dir_entry.get().strip()
        self.output_excel = self.output_entry.get().strip()

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

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        self.image_files = sorted([f for f in os.listdir(self.image_dir)
                                   if f.lower().endswith(valid_exts)])
        if not self.image_files:
            messagebox.showerror("Error", "No images found in the image directory.")
            return

        self.current_index = 0
        self.comments = {}
        self.load_new_image()

        # Populate the heat map class combobox.
        self.class_name_to_id = {}
        if self.categories:
            class_names = []
            for cat_id, cat_name in self.categories.items():
                class_names.append(cat_name)
                self.class_name_to_id[cat_name] = cat_id
            self.heatmap_class_combobox['values'] = class_names
            if class_names:
                self.heatmap_class_combobox.current(0)

    def load_new_image(self):
        """Load a new image (for next/prev) and reset zoom and pan."""
        if not self.image_files:
            return

        current_file = self.image_files[self.current_index]
        image_path = os.path.join(self.image_dir, current_file)
        self.base_cv_image = cv2.imread(image_path)
        if self.base_cv_image is None:
            messagebox.showerror("Error", f"Failed to load image: {current_file}")
            return

        self.highlighted_annotation_index = None  # reset highlight

        # Reset zoom factor based on image size.
        orig_h, orig_w = self.base_cv_image.shape[:2]
        base_zoom = min(self.canvas_width / orig_w, self.canvas_height / orig_h, 1.0)
        self.zoom_factor = base_zoom
        new_w = int(orig_w * self.zoom_factor)
        new_h = int(orig_h * self.zoom_factor)
        self.pan_x = (self.canvas_width - new_w) // 2
        self.pan_y = (self.canvas_height - new_h) // 2

        self.update_annotation_list()
        self.refresh_image()

        self.comment_text.delete("1.0", tk.END)
        if self.image_files[self.current_index] in self.comments:
            self.comment_text.insert(tk.END, self.comments[self.image_files[self.current_index]])

    def refresh_image(self):
        """Redraw the current image with current zoom/pan and annotation toggle (without resetting them)."""
        if self.base_cv_image is None:
            return

        img = self.base_cv_image.copy()
        current_file = self.image_files[self.current_index]
        if self.annotations_on.get() and current_file in self.annotations_by_filename:
            ann_list = self.annotations_by_filename[current_file]
            img = draw_annotations(img, ann_list, self.categories, self.category_colors,
                                   highlighted_index=self.highlighted_annotation_index)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img)

        orig_w, orig_h = pil_image.size
        new_w = int(orig_w * self.zoom_factor)
        new_h = int(orig_h * self.zoom_factor)
        zoomed_image = pil_image.resize((new_w, new_h), Image.ANTIALIAS)
        self.photo_image = ImageTk.PhotoImage(zoomed_image)

        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.photo_image)

    def update_annotation_list(self):
        """Update the annotation list panel for the current image."""
        self.annotation_listbox.delete(0, tk.END)
        current_file = self.image_files[self.current_index]
        if current_file in self.annotations_by_filename:
            for i, ann in enumerate(self.annotations_by_filename[current_file]):
                cat_id = ann.get("category_id")
                cat_label = self.categories.get(cat_id, str(cat_id))
                self.annotation_listbox.insert(tk.END, f"{i + 1}: {cat_label}")
        else:
            self.annotation_listbox.insert(tk.END, "No annotations")

    def on_annotation_select(self, event):
        """When an annotation is selected in the list, highlight it in the image."""
        selection = self.annotation_listbox.curselection()
        if selection:
            self.highlighted_annotation_index = selection[0]
        else:
            self.highlighted_annotation_index = None
        self.refresh_image()

    def on_mouse_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.orig_pan_x = self.pan_x
        self.orig_pan_y = self.pan_y

    def on_mouse_drag(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x = self.orig_pan_x + dx
        self.pan_y = self.orig_pan_y + dy
        self.refresh_image()

    def on_mouse_wheel(self, event):
        if hasattr(event, 'delta'):
            factor = 1.1 if event.delta > 0 else 0.9
        elif event.num == 4:
            factor = 1.1
        elif event.num == 5:
            factor = 0.9
        else:
            factor = 1.0

        new_zoom = self.zoom_factor * factor
        new_zoom = max(0.1, min(new_zoom, 10.0))
        factor = new_zoom / self.zoom_factor

        self.pan_x = event.x - (event.x - self.pan_x) * factor
        self.pan_y = event.y - (event.y - self.pan_y) * factor
        self.zoom_factor = new_zoom
        self.update_zoom_label()
        self.refresh_image()

    def on_canvas_configure(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.refresh_image()

    def zoom_in(self):
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

    def update_zoom_label(self):
        self.zoom_label.config(text=f"Zoom: {self.zoom_factor:.2f}x")

    def save_comment(self):
        current_file = self.image_files[self.current_index]
        comment = self.comment_text.get("1.0", tk.END).strip()
        self.comments[current_file] = comment

    def next_image(self):
        self.save_comment()
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.load_new_image()

    def prev_image(self):
        self.save_comment()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_new_image()

    def toggle_annotations(self):
        self.refresh_image()
        self.update_annotation_list()

    def on_key(self, event):
        if event.keysym == "Left":
            self.prev_image()
        elif event.keysym == "Right":
            self.next_image()
        elif event.char.lower() == "a":
            self.toggle_annotations()
        elif event.keysym in ("plus", "equal"):
            self.zoom_in()
        elif event.keysym in ("minus", "KP_Subtract"):
            self.zoom_out()

    def export_reviews(self):
        self.save_comment()
        data = []
        for image_file in self.image_files:
            data.append({"image_name": image_file, "comment": self.comments.get(image_file, "")})
        df = pd.DataFrame(data)
        try:
            df.to_excel(self.output_excel, index=False)
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
                cat_id = ann.get("category_id")
                per_class_counts[cat_id] = per_class_counts.get(cat_id, 0) + 1
        avg_annotations = total_annotations / total_images if total_images else 0

        summary_text = (f"Total Images: {total_images}\n"
                        f"Total Annotations: {total_annotations}\n"
                        f"Avg Annotations/Image: {avg_annotations:.2f}")

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

        self.stats_summary_label.config(text=summary_text)
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        self.chart_canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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


#########################
# Main Application Entry
#########################

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewApp(root)
    root.mainloop()
