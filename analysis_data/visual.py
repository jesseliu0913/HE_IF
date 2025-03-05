import tifffile as tiff
from PIL import Image
import numpy as np

image_path = "./data/crc01/18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif"
output_path = "converted_image.png"

img_data = tiff.imread(image_path)

if img_data.ndim == 3: 
    img = Image.fromarray(img_data[:, :, 0])  
elif img_data.ndim == 2:  
    img = Image.fromarray(img_data)
else:
    raise ValueError("Unexpected image format.")
img.save(output_path)

print(f"Image saved successfully: {output_path}")



