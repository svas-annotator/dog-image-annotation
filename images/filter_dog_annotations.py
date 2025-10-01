import pandas as pd
import os
import cv2
import xml.etree.ElementTree as ET

# 🔧 Step 1: Set your paths
image_folder = r"C:\Users\Admin\dog-image-annotation\images"
annotations_file = r"C:\Users\Admin\dog-image-annotation\images\train-annotations-bbox.csv"
class_file = r"C:\Users\Admin\dog-image-annotation\images\class-descriptions-boxable.csv"
output_file = r"C:\Users\Admin\dog-image-annotation\dog_annotations_filtered.csv"

# 🐾 Step 2: Get LabelName for "Dog"
class_df = pd.read_csv(class_file, header=None)
dog_label = class_df[class_df[1] == "Dog"].iloc[0, 0]  # e.g., /m/0bt9lr

# 📸 Step 3: Get ImageIDs from downloaded images
image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.endswith(".jpg")]

# 📄 Step 4: Load and filter annotations
chunksize = 100000  # Process in chunks to avoid memory issues
filtered_rows = []

for chunk in pd.read_csv(annotations_file, chunksize=chunksize):
    dog_rows = chunk[(chunk["LabelName"] == dog_label) & (chunk["ImageID"].isin(image_ids))]
    filtered_rows.append(dog_rows)

# 🧾 Step 5: Combine and save
final_df = pd.concat(filtered_rows)
final_df.to_csv(output_file, index=False)

print(f"✅ Done! Filtered annotations saved to:\n{output_file}")