import tifffile as tiff
import matplotlib.pyplot as plt
import os, cv2


# file_path = "./data/crc01/18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif"
# output_dir = "./plot/"
# os.makedirs(output_dir, exist_ok=True)

# img = tiff.imread(file_path)
# print(f"Loaded Image Shape: {img.shape}, Data Type: {img.dtype}")  

# scale_percent = 10  
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# save_path = os.path.join(output_dir, "color_preview.png")
# plt.figure(figsize=(8, 8))
# plt.imshow(resized_img)  
# plt.title("Color H&E Image")
# plt.axis("off")
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.close()

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

file_path = "./data/crc01/P37_S29_A24_C59kX_E15_20220106_014304_946511-zlib.ome.tiff"
output_dir = "./plot/orion_if_channels/"
os.makedirs(output_dir, exist_ok=True)

img = tiff.imread(file_path)
print(f"Loaded Image Shape: {img.shape}, Data Type: {img.dtype}")

for i in range(img.shape[0]):  
    channel_img = img[i].astype(np.float32)

    channel_img = (channel_img - np.min(channel_img)) / (np.max(channel_img) - np.min(channel_img) + 1e-6) * 255
    channel_img = channel_img.astype(np.uint8)

    scale_percent = 10
    width = int(channel_img.shape[1] * scale_percent / 100)
    height = int(channel_img.shape[0] * scale_percent / 100)
    resized_img = cv2.resize(channel_img, (width, height), interpolation=cv2.INTER_AREA)

    save_path = os.path.join(output_dir, f"channel_{i}.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(resized_img, cmap="gray")
    plt.title(f"Channel {i}")
    plt.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved Channel {i} preview: {save_path}")



