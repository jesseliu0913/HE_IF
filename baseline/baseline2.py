import os
import tifffile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

class HEBiomarkerDataset(Dataset):
    def __init__(self, df, he_image, biomarkers, morphology_features):
        self.df = df
        self.he_image = he_image
        self.biomarkers = biomarkers
        
        # Check which morphology features actually exist in the dataframe
        available_features = []
        for feature in morphology_features:
            if feature in df.columns:
                available_features.append(feature)
        
        if not available_features:
            print("Warning: No morphology features found in dataset! Using only Area and centroid coordinates.")
            self.morphology_features = ["Area", "X_centroid", "Y_centroid"]
        else:
            print(f"Using {len(available_features)} morphology features: {available_features}")
            self.morphology_features = available_features
            
        # Create a scaler for morphology features
        self.scaler = StandardScaler()
        morphology_data = df[self.morphology_features].values
        self.scaler.fit(morphology_data)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y, area = int(row["X_centroid"]), int(row["Y_centroid"]), int(row["Area"])
        radius = int(np.sqrt(area / np.pi))
        
        # Extract and normalize RGB patch
        x_min, x_max = max(0, x - radius), min(self.he_image.shape[1], x + radius)
        y_min, y_max = max(0, y - radius), min(self.he_image.shape[0], y + radius)
        
        if x_max <= x_min or y_max <= y_min or x_min >= self.he_image.shape[1] or y_min >= self.he_image.shape[0]:
            patch = np.zeros((16, 16, 3), dtype=np.float32)
        else:
            patch = self.he_image[y_min:y_max, x_min:x_max]
            
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                patch = np.zeros((16, 16, 3), dtype=np.float32)
            elif patch.shape[0] != 16 or patch.shape[1] != 16:
                temp_patch = np.zeros((16, 16, 3), dtype=np.float32)
                h, w = min(16, patch.shape[0]), min(16, patch.shape[1])
                temp_patch[:h, :w] = patch[:h, :w]
                patch = temp_patch
        
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        
        # Extract and normalize morphology features
        morphology_values = [row[feature] for feature in self.morphology_features]
        morphology_values = np.array(morphology_values, dtype=np.float32).reshape(1, -1)
        morphology_values = self.scaler.transform(morphology_values).flatten()
        
        # Get target biomarker values
        targets = np.zeros(len(self.biomarkers), dtype=np.float32)
        for i, biomarker in enumerate(self.biomarkers):
            targets[i] = np.log1p(row[biomarker])
        
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(morphology_values, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

class BiomarkerMLP(nn.Module):
    def __init__(self, num_biomarkers, num_morphology_features):
        super(BiomarkerMLP, self).__init__()
        
        # Input is flattened RGB (3*16*16) + morphology features
        rgb_input_size = 3 * 16 * 16
        total_input_size = rgb_input_size + num_morphology_features
        
        self.fc = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_biomarkers)
        )
        
    def forward(self, rgb_input, morphology_input):
        # Flatten RGB input
        rgb_flat = rgb_input.view(rgb_input.size(0), -1)
        
        # Concatenate RGB and morphology features
        combined_input = torch.cat((rgb_flat, morphology_input), dim=1)
        
        # Pass through MLP
        x = self.fc(combined_input)
        return x

def train_evaluation_pipeline(data_directory, biomarkers, morphology_features, num_epochs=5):
    folders = [os.path.join(data_directory, d) for d in os.listdir(data_directory) 
               if os.path.isdir(os.path.join(data_directory, d))]
    
    random.seed(42)
    random.shuffle(folders)
    
    train_folders = folders[:8]
    val_folders = [folders[8]]
    test_folders = [folders[9]]
    
    # Print info about the requested morphology features
    print(f"Requested morphology features: {morphology_features}")
    print("Note: Only features present in the CSV files will be used.")
    
    # First, determine which morphology features are available across all datasets
    common_features = set()
    first = True
    
    # Check train folders
    for folder in train_folders + val_folders + test_folders:
        file_index = os.path.basename(folder)
        csv_file = None
        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith('.csv') and file_index in file:
                csv_file = file_path
                break
        
        if csv_file is None:
            continue
            
        df = pd.read_csv(csv_file)
        available = [f for f in morphology_features if f in df.columns]
        
        if first:
            common_features = set(available)
            first = False
        else:
            common_features = common_features.intersection(set(available))
    
    # Ensure we have at least these basic features
    required_features = ["Area", "X_centroid", "Y_centroid"]
    for feature in required_features:
        if feature not in common_features:
            common_features.add(feature)
    
    # Convert set back to list
    common_morphology_features = sorted(list(common_features))
    print(f"Common morphology features across all datasets: {common_morphology_features}")
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for folder in train_folders:
        file_index = os.path.basename(folder)
        csv_file = None
        he_path = None
        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith('.csv') and file_index in file:
                csv_file = file_path
            elif 'registered.ome.tif' in file:
                he_path = file_path
        
        if csv_file is None or he_path is None:
            print(f"Missing files for {folder}")
            continue
        
        print(f"Processing {folder} for training...")
        df = pd.read_csv(csv_file)
        he_image = tifffile.imread(he_path)
        
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, common_morphology_features)
        train_datasets.append(dataset)
    
    for folder in val_folders:
        file_index = os.path.basename(folder)
        csv_file = None
        he_path = None
        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith('.csv') and file_index in file:
                csv_file = file_path
            elif 'registered.ome.tif' in file:
                he_path = file_path
        
        if csv_file is None or he_path is None:
            print(f"Missing files for {folder}")
            continue
        
        print(f"Processing {folder} for validation...")
        df = pd.read_csv(csv_file)
        he_image = tifffile.imread(he_path)
        
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, common_morphology_features)
        val_datasets.append(dataset)
    
    test_dfs = []
    
    for folder in test_folders:
        file_index = os.path.basename(folder)
        csv_file = None
        he_path = None
        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith('.csv') and file_index in file:
                csv_file = file_path
            elif 'registered.ome.tif' in file:
                he_path = file_path
        
        if csv_file is None or he_path is None:
            print(f"Missing files for {folder}")
            continue
        
        print(f"Processing {folder} for testing...")
        df = pd.read_csv(csv_file)
        he_image = tifffile.imread(he_path)
        
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, common_morphology_features)
        test_datasets.append(dataset)
        test_dfs.append((df, file_index))
    
    if not train_datasets or not val_datasets or not test_datasets:
        print("Missing required datasets")
        return
    
    # Get the actual number of morphology features used
    num_morphology_features = len(common_morphology_features)
    print(f"Using {num_morphology_features} morphology features")
    
    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_datasets), 
                             batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_datasets),
                           batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(torch.utils.data.ConcatDataset(test_datasets),
                            batch_size=256, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiomarkerMLP(len(biomarkers), num_morphology_features).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    best_val_loss = float('inf')
    
    print(f"Training model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for rgb_inputs, morph_inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            rgb_inputs, morph_inputs, targets = rgb_inputs.to(device), morph_inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(rgb_inputs, morph_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * rgb_inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for rgb_inputs, morph_inputs, targets in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                rgb_inputs, morph_inputs, targets = rgb_inputs.to(device), morph_inputs.to(device), targets.to(device)
                outputs = model(rgb_inputs, morph_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * rgb_inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers,
                'morphology_features': common_morphology_features
            }, os.path.join(data_directory, "model_results", "best_biomarker_model.pt"))
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    checkpoint = torch.load(os.path.join(data_directory, "model_results", "best_biomarker_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for rgb_inputs, morph_inputs, targets in tqdm(test_loader, desc="Testing"):
            rgb_inputs, morph_inputs, targets = rgb_inputs.to(device), morph_inputs.to(device), targets.to(device)
            outputs = model(rgb_inputs, morph_inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * rgb_inputs.size(0)
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")
    
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    results = {}
    
    for i, biomarker in enumerate(biomarkers):
        pearson_r, _ = pearsonr(all_targets[:, i], all_outputs[:, i])
        spearman_r, _ = spearmanr(all_targets[:, i], all_outputs[:, i])
        
        actual_orig = np.expm1(all_targets[:, i])
        pred_orig = np.expm1(np.clip(all_outputs[:, i], -10, 10))
        c_index = concordance_index(actual_orig, pred_orig)
        
        results[biomarker] = {
            'PearsonR': pearson_r,
            'SpearmanR': spearman_r,
            'C-index': c_index
        }
    
    results_df = pd.DataFrame(results).T
    
    output_dir = os.path.join(data_directory, "model_results_baseline2")
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics_5.csv"))
    
    test_df, test_idx = test_dfs[0]
    
    for i, biomarker in enumerate(biomarkers):
        clipped_preds = np.clip(all_outputs[:, i], -10, 10)
        test_df[f"{biomarker}_predicted"] = np.expm1(clipped_preds)
    
    test_df.to_csv(os.path.join(output_dir, f"{test_idx}_predictions_5.csv"), index=False)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'biomarkers': biomarkers,
        'morphology_features': common_morphology_features
    }, os.path.join(output_dir, "biomarker_model_5.pt"))
    
    print(f"Results saved to {output_dir}")
    return results_df

if __name__ == "__main__":
    data_directory = "../data/data"
    biomarkers = [
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
        "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"
    ]
    
    morphology_features = [
        "Area", "Perimeter", "Circularity", "Eccentricity", 
        "X_centroid", "Y_centroid", "MajorAxis", "MinorAxis"
    ]
    
    train_evaluation_pipeline(data_directory, biomarkers, morphology_features, num_epochs=5)