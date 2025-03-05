import tifffile as tiff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from concurrent.futures import ThreadPoolExecutor


image_path = "./data/crc01/P37_S29_A24_C59kX_E15_20220106_014304_946511-zlib.ome.tiff"
markers_csv_path = "./data/crc01/markers.csv"

img = tiff.imread(image_path)  
print(f"Loaded IF Image Shape: {img.shape}, Data Type: {img.dtype}")


markers_df = pd.read_csv(markers_csv_path)
markers = markers_df.iloc[:, 0].tolist()  

output_dir = "./plot/if_markers/"
os.makedirs(output_dir, exist_ok=True)


scale_factor = 0.1  
H, W = int(img.shape[1] * scale_factor), int(img.shape[2] * scale_factor)
img_resized = np.array([cv2.resize(img[i], (W, H), interpolation=cv2.INTER_AREA) for i in range(img.shape[0])])

print(f"Downsampled Image Shape: {img_resized.shape}")

def normalize_img(channel_img):
    """Normalize image to [0,255] using OpenCV for fast processing"""
    channel_img = channel_img.astype(np.float32)
    return cv2.normalize(channel_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def enhance_contrast(channel_img):
    """Apply log transformation and CLAHE for better contrast visualization."""
    # Convert to float32
    channel_img = channel_img.astype(np.float32)
    
    # Apply log transformation (to enhance weak signals)
    channel_img = np.log1p(channel_img)

    # Normalize to 0-255
    channel_img = (channel_img - np.min(channel_img)) / (np.max(channel_img) - np.min(channel_img) + 1e-6) * 255
    channel_img = channel_img.astype(np.uint8)

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(channel_img)

    return enhanced_img


def process_marker(i, marker):
    if i < img_resized.shape[0]:  
        marker_img = enhance_contrast(img_resized[i])

        heatmap = cv2.applyColorMap(marker_img, cv2.COLORMAP_HOT)
        save_path = os.path.join(output_dir, f"{marker}_expression.png")
        cv2.imwrite(save_path, heatmap)

        print(f"✅ Saved {marker} expression heatmap")
    else:
        print(f"⚠️ Marker {marker} index {i} is out of bounds")

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_marker, range(len(markers)), markers)

plt.figure(figsize=(10, 6))
for i, marker in enumerate(markers):
    if i < img_resized.shape[0]:  
        marker_img = img_resized[i].flatten()
        plt.hist(marker_img, bins=50, alpha=0.5, label=marker)

plt.xlabel("Intensity")
plt.ylabel("Pixel Count")
plt.title("Fluorescence Intensity Distribution of Markers")
plt.legend()
plt.savefig("./plot/marker_intensity_histogram.png", dpi=300)
plt.close()

print("✅ Saved fluorescence intensity histogram: ./plot/marker_intensity_histogram.png")

print("🎉 **Optimized Analysis Complete!** Check `./plot/` for results.")

