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
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

class HEBiomarkerDataset(Dataset):
    def __init__(self, df, he_image, biomarkers, transform=None):
        self.df = df
        self.he_image = he_image
        self.biomarkers = biomarkers
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y = int(row["X_centroid"]), int(row["Y_centroid"])
        
        patch_size = 224
        half_size = patch_size // 2
        
        x_min, x_max = max(0, x - half_size), min(self.he_image.shape[1], x + half_size)
        y_min, y_max = max(0, y - half_size), min(self.he_image.shape[0], y + half_size)
        
        if x_max <= x_min or y_max <= y_min or x_min >= self.he_image.shape[1] or y_min >= self.he_image.shape[0]:
            patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        else:
            patch_height = y_max - y_min
            patch_width = x_max - x_min
            
            patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
            
            extract = self.he_image[y_min:y_max, x_min:x_max]
            
            y_offset = (patch_size - patch_height) // 2
            x_offset = (patch_size - patch_width) // 2
            
            patch[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width] = extract
        
        patch = patch.astype(np.float32) / 255.0
        
        if self.transform:
            patch = self.transform(patch)
        else:
            patch = np.transpose(patch, (2, 0, 1))
            patch = torch.tensor(patch, dtype=torch.float32)
        
        targets = np.zeros(len(self.biomarkers), dtype=np.float32)
        for i, biomarker in enumerate(self.biomarkers):
            targets[i] = np.log1p(row[biomarker])
        
        return patch, torch.tensor(targets, dtype=torch.float32)

class BiomarkerViT(nn.Module):
    def __init__(self, num_biomarkers, pretrained=True):
        super(BiomarkerViT, self).__init__()
        
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16()
        
        vit_output_size = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        
        self.regression_head = nn.Sequential(
            nn.Linear(vit_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_biomarkers)
        )
        
    def forward(self, x):
        features = self.vit(x)
        return self.regression_head(features)

def train_regression_only_pipeline(data_directory, biomarkers, batch_size=1024, num_epochs=5):
    folders = [os.path.join(data_directory, d) for d in os.listdir(data_directory) 
               if os.path.isdir(os.path.join(data_directory, d))]

    output_directory = "./results"
    random.seed(42)
    random.shuffle(folders)
    
    train_folders = folders[:8]
    val_folders = [folders[8]] if len(folders) > 8 else [folders[0]]
    test_folders = [folders[9]] if len(folders) > 9 else [folders[0]]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_datasets = []
    val_datasets = []
    test_datasets = []
    test_dfs = []
    
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
        
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, transform=transform)
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
        
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, transform=transform)
        val_datasets.append(dataset)
    
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
        
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, transform=transform)
        test_datasets.append(dataset)
        test_dfs.append((df, file_index))
    
    if not train_datasets or not val_datasets or not test_datasets:
        print("Missing required datasets")
        return
    
    train_loader = DataLoader(torch.utils.data.ConcatDataset(train_datasets), 
                             batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(torch.utils.data.ConcatDataset(val_datasets),
                           batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(torch.utils.data.ConcatDataset(test_datasets),
                            batch_size=batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiomarkerViT(len(biomarkers), pretrained=True).to(device)
    

    for name, param in model.named_parameters():
        if 'vit' in name:  
            param.requires_grad = False
        else:  
            param.requires_grad = True
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    output_dir = os.path.join(output_directory, "model_results_vit_regression")
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"Training regression head for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers
            }, os.path.join(output_dir, "best_biomarker_vit_regression.pt"))
    
    checkpoint = torch.load(os.path.join(output_dir, "best_biomarker_vit_regression.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            
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
    
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics_vit.csv"))
    
    test_df, test_idx = test_dfs[0]
    
    for i, biomarker in enumerate(biomarkers):
        clipped_preds = np.clip(all_outputs[:, i], -10, 10)
        test_df[f"{biomarker}_predicted"] = np.expm1(clipped_preds)
    
    test_df.to_csv(os.path.join(output_dir, f"{test_idx}_predictions_vit.csv"), index=False)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'biomarkers': biomarkers
    }, os.path.join(output_dir, "biomarker_model_vit.pt"))
    
    print(f"Results saved to {output_dir}")
    return results_df

if __name__ == "__main__":
    data_directory = "../data/data"
    biomarkers = [
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
        "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"
    ]
    
    train_regression_only_pipeline(data_directory, biomarkers, batch_size=32, num_epochs=5)