import pandas as pd
import os
import cv2
import xml.etree.ElementTree as ET

# Paths
image_folder = r"C:\Users\Admin\dog-image-annotation\images"
annotation_file = r"C:\Users\Admin\dog-image-annotation\images\dog_annotations_filtered.csv"
output_folder = r"C:\Users\Admin\dog-image-annotation\annotated_images"

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

    # Filter annotations for this image
    boxes = df[df["ImageID"] == image_id]

    for _, row in boxes.iterrows():
        # Convert normalized coordinates to pixel values
        xmin = int(row["XMin"] * width)
        xmax = int(row["XMax"] * width)
        ymin = int(row["YMin"] * height)
        ymax = int(row["YMax"] * height)

        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, "Dog", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save annotated image
    cv2.imwrite(os.path.join(output_folder, f"{image_id}_annotated.jpg"), image)

print("✅ Bounding boxes visualized and saved to:", output_folder)