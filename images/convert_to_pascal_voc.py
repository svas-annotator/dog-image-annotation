import pandas as pd
import os
import cv2
import xml.etree.ElementTree as ET

# ✅ Paths
image_folder = r"C:\Users\Admin\dog-image-annotation\images"
annotation_file = r"C:\Users\Admin\dog-image-annotation\images\dog_annotations_filtered.csv"
output_folder = r"C:\Users\Admin\dog-image-annotation\xml_annotations"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load annotations
df = pd.read_csv(annotation_file)

# Loop through each image
for image_id in df["ImageID"].unique():
    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        continue

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    boxes = df[df["ImageID"] == image_id]

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = "images"
    ET.SubElement(annotation, "filename").text = f"{image_id}.jpg"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for _, row in boxes.iterrows():
        xmin = int(row["XMin"] * width)
        xmax = int(row["XMax"] * width)
        ymin = int(row["YMin"] * height)
        ymax = int(row["YMax"] * height)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "Dog"
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(row.get("IsTruncated", 0))
        ET.SubElement(obj, "difficult").text = "0"

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(output_folder, f"{image_id}.xml"))

print("✅ Pascal VOC XML files saved to:", output_folder)