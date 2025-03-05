import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd


image_path = "./data/crc01/P37_S29_A24_C59kX_E15_20220106_014304_946511-zlib.ome.tiff"
markers_csv_path = "./data/crc01/markers.csv"

img = tiff.imread(image_path)  
print(f"Loaded IF Image Shape: {img.shape}, Data Type: {img.dtype}")
markers_df = pd.read_csv(markers_csv_path)
markers = markers_df.iloc[:, 0].tolist()  

output_dir = "./plot/marker_overlays/"
os.makedirs(output_dir, exist_ok=True)

import cv2
import numpy as np

def enhance_contrast(channel_img):
    """Apply CLAHE and gamma correction for better contrast visualization."""
    channel_img = channel_img.astype(np.float32)
    
    gamma = 2.0  
    channel_img = np.power(channel_img / np.max(channel_img), gamma) * 255

    channel_img = channel_img.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(channel_img)

    return enhanced_img

def create_overlay(marker_1_idx, marker_2_idx, marker_1_name, marker_2_name, color1=(255, 0, 0), color2=(0, 255, 0)):
    """Overlay two marker channels with custom colors"""
    if marker_1_idx < img.shape[0] and marker_2_idx < img.shape[0]:
        marker_1_img = enhance_contrast(img[marker_1_idx])  
        marker_2_img = enhance_contrast(img[marker_2_idx])  

        overlay_img = np.zeros((marker_1_img.shape[0], marker_1_img.shape[1], 3), dtype=np.uint8)
        overlay_img[..., 0] = marker_1_img * (color1[0] / 255)  
        overlay_img[..., 1] = marker_2_img * (color2[1] / 255) 

        save_path = os.path.join(output_dir, f"{marker_1_name}_{marker_2_name}_overlay.png")
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay_img)
        plt.title(f"{marker_1_name} (Red) & {marker_2_name} (Green) Co-localization")
        plt.axis("off")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✅ Saved {marker_1_name} & {marker_2_name} overlay: {save_path}")

if "CD3e" in markers and "CD8a" in markers:
    cd3_idx = markers.index("CD3e")
    cd8_idx = markers.index("CD8a")
    create_overlay(cd3_idx, cd8_idx, "CD3e", "CD8a", color1=(255, 0, 0), color2=(0, 255, 0))  

if "CD3e" in markers and "FOXP3" in markers:
    cd3_idx = markers.index("CD3e")
    foxp3_idx = markers.index("FOXP3")
    create_overlay(cd3_idx, foxp3_idx, "CD3e", "FOXP3", color1=(255, 0, 0), color2=(0, 255, 255))  

if "CD3e" in markers and "Pan-CK" in markers:
    cd3_idx = markers.index("CD3e")
    panck_idx = markers.index("Pan-CK")
    create_overlay(cd3_idx, panck_idx, "CD3e", "Pan-CK", color1=(255, 0, 0), color2=(255, 255, 0))  

print("Marker Overlays Complete!!!")


