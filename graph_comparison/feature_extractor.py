import os
import cv2
import json
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from sklearn.model_selection import train_test_split

def extract_patch(he_image, x, y, area, size=224):
    image = np.transpose(he_image, (1, 2, 0)) 
    radius = int(np.sqrt(area / np.pi))

    x_min, x_max = max(0, x - radius), min(image.shape[1], x + radius)
    y_min, y_max = max(0, y - radius), min(image.shape[0], y + radius)
    
    if x_max <= x_min or y_max <= y_min or x_min >= image.shape[1] or y_min >= image.shape[0]:
        patch = np.zeros((size, size, 3), dtype=np.float32)
    else:
        patch = image[y_min:y_max, x_min:x_max]
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

    return patch

def process_batch(batch_df, he_image, gpu_id, output_file):
    device = torch.device(f"cuda:{gpu_id}")
    
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
    model.eval()
    
    results = []
    
    for _, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
        cell_id = row.name
        x, y, area = int(row['X']), int(row['Y']), int(row['AREA'])
        
        patch = extract_patch(he_image, x, y, area)
        pil_image = transforms.ToPILImage()(patch)
        
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        results.append({
            'cell_id': cell_id,
            'x': x,
            'y': y,
            'features': features.tolist()
        })
    
    with open(output_file, 'w') as f:
        json.dump(results, f)

def main():
    DATA_PATH = "/playpen/jesse/HIPI/preprocess/data"
    OUTPUT_PATH = "./cell_feature"
    he_path = "/playpen/jesse/HIPI/preprocess/data/CRC03-HE.ome.tif"
    csv_file = "/playpen/jesse/HIPI/preprocess/data/CRC03_new_coordinates.csv"
    features_dir = os.path.join(DATA_PATH, "cell_features")
    
    os.makedirs(features_dir, exist_ok=True)
    
    he_image = tifffile.imread(he_path)
    df = pd.read_csv(csv_file)
    
    num_gpus = torch.cuda.device_count()
    
    batches = np.array_split(df, num_gpus)
    
    processes = []
    for i, batch_df in enumerate(batches):
        output_file = os.path.join(features_dir, f"features_batch_{i}.json")
        p = mp.Process(target=process_batch, args=(batch_df, he_image, i, output_file))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
    
    feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir) if f.startswith("features_batch_")]
    combined_features = {}
    
    for file in feature_files:
        with open(file, 'r') as f:
            batch_data = json.load(f)
            for item in batch_data:
                combined_features[item['cell_id']] = {
                    'x': item['x'],
                    'y': item['y'],
                    'features': item['features']
                }
    
    features_output = os.path.join(OUTPUT_PATH, "cell_features.npz")
    
    cell_ids = list(combined_features.keys())
    features_array = np.array([combined_features[cell_id]['features'] for cell_id in cell_ids])
    coords_array = np.array([(combined_features[cell_id]['x'], combined_features[cell_id]['y']) for cell_id in cell_ids])
    
    np.savez_compressed(
        features_output,
        cell_ids=np.array(cell_ids),
        features=features_array,
        coords=coords_array
    )
    
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    data_splits = {
        'train': train_df.index.tolist(),
        'val': val_df.index.tolist(),
        'test': test_df.index.tolist()
    }
    
    with open(os.path.join(OUTPUT_PATH, "data_splits.json"), 'w') as f:
        json.dump(data_splits, f)
    
    print(f"Features saved to {features_output}")
    print(f"Data splits saved to {os.path.join(OUTPUT_PATH, 'data_splits.json')}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()


# CUDA_VISIBLE_DEVICES=4,5,7 python feature_extractor.py