import os
import cv2
import json
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_patch_mean_channels(he_image, x, y, area, target_size=(28, 28)):
    radius = int(np.sqrt(area / np.pi))
    
    _, height, width = he_image.shape
    
    x_min, x_max = max(0, x - radius), min(width, x + radius)
    y_min, y_max = max(0, y - radius), min(height, y + radius)
    
    if x_max <= x_min or y_max <= y_min or x_min >= width or y_min >= height:
        return np.zeros(target_size, dtype=np.float32)
    else:
        patch = he_image[:, y_min:y_max, x_min:x_max]
        mean_patch = np.mean(patch, axis=0)
        
        if mean_patch.shape[0] > 0 and mean_patch.shape[1] > 0:
            resized_patch = cv2.resize(mean_patch, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized_patch = np.zeros(target_size, dtype=np.float32)
            
        return resized_patch

def process_cells(df, he_image, output_path, target_size=(28, 28)):
    results = {}
    
    # Initialize a 3D array to store all features (num_cells, height, width)
    num_cells = len(df)
    features_array = np.zeros((num_cells, target_size[0], target_size[1]), dtype=np.float32)
    coords_array = np.zeros((num_cells, 2), dtype=np.int32)
    cell_ids = []
    
    for i, (cell_id, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Processing cells")):
        x, y, area = int(row['hne_X']), int(row['hne_Y']), int(row['AREA'])
        
        # Extract and resize patch
        mean_patch = extract_patch_mean_channels(he_image, x, y, area, target_size)
        
        # Store in arrays
        features_array[i] = mean_patch
        coords_array[i] = [x, y]
        cell_ids.append(cell_id)
    
    os.makedirs(output_path, exist_ok=True)
    features_output = os.path.join(output_path, "cell_features.npz")
    
    np.savez_compressed(
        features_output,
        cell_ids=np.array(cell_ids),
        features=features_array,
        coords=coords_array
    )
    
    return features_output

def main():
    DATA_PATH = "/playpen/jesse/HIPI/preprocess/data"
    OUTPUT_PATH = "./cell_feature_naive"
    he_path = "/playpen/jesse/HIPI/preprocess/data/CRC03-HE.ome.tif"
    csv_file = "/playpen/jesse/HIPI/preprocess/data/CRC03_new_coordinates.csv"
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    he_image = tifffile.imread(he_path)
    df = pd.read_csv(csv_file)
    
    features_output = process_cells(df, he_image, OUTPUT_PATH)
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    data_splits = {
        'train': train_df.index.tolist(),
        'val': val_df.index.tolist(),
        'test': test_df.index.tolist()
    }
    
    splits_path = os.path.join(OUTPUT_PATH, "data_splits.json")
    with open(splits_path, 'w') as f:
        json.dump(data_splits, f)
    
    print(f"Features saved to {features_output}")
    print(f"Data splits saved to {splits_path}")

if __name__ == "__main__":
    main()