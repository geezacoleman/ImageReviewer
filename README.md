# Image Review Tool

A Python-based desktop application for reviewing annotated images. It supports multiple annotation formats (COCO, PASCAL
VOC, YOLO, and WeedCOCO) and displays them over images in a practical three-pane layout. Built with Tkinter, it includes
features to streamline the review process and analyze datasets effectively.

![image](https://github.com/user-attachments/assets/44504f78-1c0e-44f0-abf8-4f64ace95451)

## Features

- **Annotation Format Support:**  
  Loads annotations from various formats:
    - **COCO:** JSON files with segmentation masks or bounding boxes.
    - **PASCAL VOC:** XML files with bounding box annotations.
    - **YOLO:** Text files with bounding boxes, dynamically reading image dimensions; supports an optional `labels.txt`
      for class names.
    - **WeedCOCO:** An extended COCO format with additional metadata like agricultural contexts, shown in a dedicated
      tab.

- **Annotation Display:**  
  Overlays annotations—bounding boxes or segmentation masks—on images, depending on what’s provided. Highlights specific
  annotations when selected.

- **Filtered Cutouts:**  
  Filters images by annotation class and displays cropped thumbnails of annotated regions. Thumbnails are cached for
  faster loading, and resizing is optimized to keep the interface responsive.

- **Statistics and Heat Maps:**
    - Bar charts summarize annotation counts per class, with interactive Matplotlib controls.
    - Heat maps show the spatial distribution of annotations across images, useful for spotting patterns.

- **User Interface:**
    - **Left Panel:** Lists annotations for the current image, with a filter option.
    - **Center Panel:** Displays the main image with navigation, zoom controls, and a persistent comment section.
    - **Right Panel:** Contains tabs:
        - **Load Data:** Set image/annotation directories, output file, and annotation type.
        - **Filtered:** View class-specific cutouts.
        - **Comments:** Add and review notes per image.
        - **Stats:** Generate annotation statistics.
        - **Heat Map:** Visualize annotation placement.
        - **Metadata:** Display WeedCOCO-specific details (e.g., agricultural context).

- **Persistent Settings and Comments:**  
  Saves configuration (directories and output file) to a settings file in your home directory. Comments are stored in a
  temporary JSON file and persist across sessions, linked to the dataset.

- **Export Functionality:**  
  Exports comments, annotation counts, and class details to an Excel file for further analysis.

## Installation

### Requirements

- Python 3.x
- Tkinter (typically included with Python)
- `opencv-python`
- `Pillow`
- `pandas`
- `matplotlib`
- `numpy`

### Install Dependencies

```bash
pip install opencv-python Pillow pandas matplotlib numpy
