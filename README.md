# Image Review Tool

A Python-based desktop application designed to help you review and analyze annotated images. Built with Tkinter and optimized with performance improvements, this tool loads COCO-format JSON annotations, overlays them on images, and provides interactive features like filtered cutouts, statistics, and heat maps. The modern, three-pane layout makes navigation and review both efficient and user-friendly.

---

## Key Features

- **Image & Annotation Review:**  
  Load images along with their corresponding COCO-format JSON annotations. View images with overlaid segmentation masks or bounding boxes and highlight specific annotations.

- **Filtered View with Cutouts:**  
  Dynamically filter and display cropped cutouts based on a selected annotation class. Thumbnails are cached and the layout is optimized with debounced resizing, ensuring smooth performance even with large datasets.

- **Interactive Statistics & Heat Maps:**  
  Generate interactive bar charts summarizing annotation counts per class. Create heat maps to visualize the spatial distribution of annotations across images, complete with embedded matplotlib navigation tools.

- **Modern, Ergonomic UI:**  
  A three-pane layout:
  - **Left Panel:** Displays the list of annotations for the current image.
  - **Center Panel:** Shows the main image with navigation and zoom controls.
  - **Right Panel:** Contains a Notebook with tabs for loading data, filtered view, comments, statistics, and heat maps.
  
  The UI is built with themed `ttk` widgets for a more modern look.

- **Persistent Settings:**  
  Save and load configuration settings (annotations directory, images directory, and output Excel file) using a file in the user's home directory. An explicit "Save Settings" button allows you to deliberately store your configuration for future sessions.

- **Performance Optimizations:**  
  - **Thumbnail Caching:** Avoids reprocessing cutouts by caching generated thumbnails.
  - **Debounced Resizing:** Reduces layout recalculations during rapid window resizes, ensuring smoother performance.

- **Export Reviews:**  
  Easily export comments and review data to an Excel file for further analysis or record-keeping.

---

## Installation

### Requirements

- Python 3.x
- Tkinter (usually comes bundled with Python)
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [Pillow](https://python-pillow.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

### Installation via pip

```bash
pip install opencv-python Pillow pandas matplotlib
