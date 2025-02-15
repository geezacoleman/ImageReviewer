import os
import json
import glob
import cv2
import argparse
import pandas as pd
import numpy as np
import colorsys


def load_annotations(annotation_dir):
    """
    Loads all COCO-format JSON files from the annotation directory.
    Returns:
      - annotations_by_filename: A dict mapping image file names to a list of annotation dicts.
      - categories: A dict mapping category_id to category name.
    """
    annotations_by_filename = {}
    categories = {}

    # Process every JSON file in the provided annotation directory.
    for json_file in glob.glob(os.path.join(annotation_dir, '*.json')):
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Build a mapping of image id to file name (from the "images" section)
        image_id_to_filename = {}
        if 'images' in data:
            for image in data['images']:
                image_id_to_filename[image['id']] = image['file_name']

        # Process each annotation: link the annotation to the image file name.
        if 'annotations' in data:
            for ann in data['annotations']:
                img_id = ann.get('image_id')
                filename = image_id_to_filename.get(img_id)
                if filename is None:
                    continue  # Skip annotations without a corresponding image.
                if filename not in annotations_by_filename:
                    annotations_by_filename[filename] = []
                annotations_by_filename[filename].append(ann)

        # Process the categories to have a mapping from category_id to category name.
        if 'categories' in data:
            for cat in data['categories']:
                cat_id = cat.get('id')
                cat_name = cat.get('name', str(cat_id))
                # If a category appears in multiple files, use the first encountered.
                if cat_id not in categories:
                    categories[cat_id] = cat_name

    return annotations_by_filename, categories


def generate_category_colors(categories):
    """
    Generate a distinct color for each category using equally spaced hues.
    Returns a dict mapping category_id to a BGR tuple.
    """
    colors = {}
    cat_ids = sorted(categories.keys())
    num_categories = len(cat_ids)
    for i, cat_id in enumerate(cat_ids):
        # Use HSV space: vary hue while keeping saturation and value at maximum.
        hue = i / num_categories
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # OpenCV uses BGR format
        color = (int(b * 255), int(g * 255), int(r * 255))
        colors[cat_id] = color
    return colors


def draw_annotations(image, annotations, categories, category_colors):
    """
    Draw segmentation annotations on the image.
    For each annotation, if segmentation data is present, each polygon is drawn in the
    class-specific color. The class name is placed inside the instance (using the bounding
    rectangle of all polygon points).
    If segmentation is absent but a bbox exists, then the bbox is drawn.
    """
    for ann in annotations:
        cat_id = ann.get('category_id')
        color = category_colors.get(cat_id, (0, 255, 0))
        label = categories.get(cat_id, str(cat_id))

        # If segmentation is available, draw the segmentation polygons.
        if 'segmentation' in ann and ann['segmentation']:
            segmentation = ann['segmentation']
            all_polygons = []

            # COCO segmentation can be a list of polygons.
            for seg in segmentation:
                if not isinstance(seg, list):
                    continue  # Skip if not a polygon list
                # Convert the flat list into a (N,2) numpy array.
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
                all_polygons.append(pts)
                # Draw the polygon outline.
                cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

            if all_polygons:
                # Combine all polygon points to compute a bounding rectangle.
                combined = np.concatenate(all_polygons, axis=0)
                x, y, w, h = cv2.boundingRect(combined)
                label_position = (x + w // 2, y + h // 2)
                # Place the class label near the center of the segmentation.
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1, cv2.LINE_AA)
        # Fallback: if no segmentation, draw the bounding box if available.
        elif 'bbox' in ann and len(ann['bbox']) == 4:
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, max(y - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description="Image Review Tool with Segmentation Support")
    parser.add_argument('--annotation_dir', required=True,
                        help="Directory containing the COCO-format annotation JSON files")
    parser.add_argument('--image_dir', required=True,
                        help="Directory containing the image files")
    parser.add_argument('--output_excel', default="image_reviews.xlsx",
                        help="Output Excel file to save the image reviews")
    args = parser.parse_args()

    # Load annotations and category information.
    annotations_by_filename, categories = load_annotations(args.annotation_dir)
    # Generate a distinct color for each category.
    category_colors = generate_category_colors(categories)

    # Get the list of images (filter by common image file extensions).
    image_files = sorted([f for f in os.listdir(args.image_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    reviews = []  # Will store dictionaries of {image_name, comment}.

    for img_file in image_files:
        img_path = os.path.join(args.image_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Unable to read image: {img_file}")
            continue

        # If annotations exist for this image, draw them.
        if img_file in annotations_by_filename:
            image = draw_annotations(image, annotations_by_filename[img_file], categories, category_colors)

        # Display the image for review.
        cv2.imshow("Image Review", image)
        print(f"\nReviewing image: {img_file}")
        # A short wait to allow the window to render.
        cv2.waitKey(1)

        # Get the reviewer’s comment from the command line.
        comment = input(f"Enter your comment for {img_file} (or leave blank to skip): ")
        reviews.append({"image_name": img_file, "comment": comment})

        # Close the current image window.
        cv2.destroyAllWindows()

    # Save the review results to an Excel file.
    df = pd.DataFrame(reviews)
    df.to_excel(args.output_excel, index=False)
    print(f"\nAll reviews have been saved to {args.output_excel}")


if __name__ == '__main__':
    main()

