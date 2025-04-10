import os
import time
import tifffile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import random


class HEBiomarkerDataset(Dataset):
    def __init__(self, df, he_image, biomarkers, morphology_features):
        start_time = time.time()
        
        self.df = df
        self.he_image = he_image
        self.biomarkers = biomarkers
        
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
            
        self.scaler = StandardScaler()
        morphology_data = df[self.morphology_features].values
        self.scaler.fit(morphology_data)
        
        end_time = time.time()
        print(f"[TIME] Dataset initialization: {end_time - start_time:.4f}s")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x, y, area = int(row["X_centroid"]), int(row["Y_centroid"]), int(row["Area"])
        radius = int(np.sqrt(area / np.pi))
        
        x_min, x_max = max(0, x - radius), min(self.he_image.shape[1], x + radius)
        y_min, y_max = max(0, y - radius), min(self.he_image.shape[0], y + radius)
        
        if x_max <= x_min or y_max <= y_min or x_min >= self.he_image.shape[1] or y_min >= self.he_image.shape[0]:
            patch = np.zeros((128, 128, 3), dtype=np.float32)
        else:
            patch = self.he_image[y_min:y_max, x_min:x_max]
            
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                patch = np.zeros((128, 128, 3), dtype=np.float32)
            elif patch.shape[0] != 128 or patch.shape[1] != 128:
                temp_patch = np.zeros((128, 128, 3), dtype=np.float32)
                h, w = min(128, patch.shape[0]), min(128, patch.shape[1])
                temp_patch[:h, :w] = patch[:h, :w]
                patch = temp_patch
        
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        
        morphology_values = [row[feature] for feature in self.morphology_features]
        morphology_values = np.array(morphology_values, dtype=np.float32).reshape(1, -1)
        morphology_values = self.scaler.transform(morphology_values).flatten()
        
        targets = np.zeros(len(self.biomarkers), dtype=np.float32)
        for i, biomarker in enumerate(self.biomarkers):
            targets[i] = np.log1p(row[biomarker])
        
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(morphology_values, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


class BiomarkerMLP(nn.Module):
    def __init__(self, num_biomarkers, num_morphology_features):
        super(BiomarkerMLP, self).__init__()
        
        rgb_input_size = 3 * 128 * 128
        total_input_size = rgb_input_size + num_morphology_features
        
        self.fc = nn.Sequential(
            nn.Linear(total_input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_biomarkers)
        )
        
    def forward(self, rgb_input, morphology_input):
        rgb_flat = rgb_input.view(rgb_input.size(0), -1)
        combined_input = torch.cat((rgb_flat, morphology_input), dim=1)
        x = self.fc(combined_input)
        return x


def load_image(file_path):
    start_time = time.time()
    image = tifffile.imread(file_path)
    end_time = time.time()
    print(f"[TIME] TIFF loading: {end_time - start_time:.4f}s for {file_path}")
    return image


def train_epoch(model, train_loader, optimizer, criterion, device, epoch_num):
    model.train()
    train_loss = 0.0
    batch_times = []
    data_transfer_times = []
    forward_times = []
    backward_times = []
    
    epoch_start = time.time()
    
    for i, (rgb_inputs, morph_inputs, targets) in enumerate(train_loader):
        batch_start = time.time()
        
        transfer_start = time.time()
        rgb_inputs = rgb_inputs.to(device)
        morph_inputs = morph_inputs.to(device)
        targets = targets.to(device)
        transfer_end = time.time()
        data_transfer_time = transfer_end - transfer_start
        data_transfer_times.append(data_transfer_time)
        
        optimizer.zero_grad()
        
        forward_start = time.time()
        outputs = model(rgb_inputs, morph_inputs)
        loss = criterion(outputs, targets)
        forward_end = time.time()
        forward_time = forward_end - forward_start
        forward_times.append(forward_time)
        
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_end = time.time()
        backward_time = backward_end - backward_start
        backward_times.append(backward_time)
        
        train_loss += loss.item() * rgb_inputs.size(0)
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: total={batch_time:.4f}s, "
                  f"transfer={data_transfer_time:.4f}s, forward={forward_time:.4f}s, "
                  f"backward={backward_time:.4f}s")
    
    train_loss /= len(train_loader.dataset)
    epoch_end = time.time()
    
    print(f"[TIME] Epoch {epoch_num} completed in {epoch_end - epoch_start:.2f}s")
    print(f"[TIME] Avg batch: {np.mean(batch_times):.4f}s, "
          f"Avg transfer: {np.mean(data_transfer_times):.4f}s, "
          f"Avg forward: {np.mean(forward_times):.4f}s, "
          f"Avg backward: {np.mean(backward_times):.4f}s")
    
    return train_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    batch_times = []
    data_transfer_times = []
    forward_times = []
    
    validate_start = time.time()
    
    with torch.no_grad():
        for rgb_inputs, morph_inputs, targets in val_loader:
            batch_start = time.time()
            
            transfer_start = time.time()
            rgb_inputs = rgb_inputs.to(device)
            morph_inputs = morph_inputs.to(device)
            targets = targets.to(device)
            transfer_end = time.time()
            data_transfer_times.append(transfer_end - transfer_start)
            
            forward_start = time.time()
            outputs = model(rgb_inputs, morph_inputs)
            loss = criterion(outputs, targets)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            val_loss += loss.item() * rgb_inputs.size(0)
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
    
    val_loss /= len(val_loader.dataset)
    validate_end = time.time()
    
    print(f"[TIME] Validation completed in {validate_end - validate_start:.2f}s")
    print(f"[TIME] Avg val batch: {np.mean(batch_times):.4f}s, "
          f"Avg val transfer: {np.mean(data_transfer_times):.4f}s, "
          f"Avg val forward: {np.mean(forward_times):.4f}s")
    
    return val_loss


def test_model(model, test_loader, criterion, device, biomarkers):
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    
    test_start = time.time()
    batch_times = []
    data_transfer_times = []
    forward_times = []
    
    with torch.no_grad():
        for rgb_inputs, morph_inputs, targets in test_loader:
            batch_start = time.time()
            
            transfer_start = time.time()
            rgb_inputs = rgb_inputs.to(device)
            morph_inputs = morph_inputs.to(device)
            targets = targets.to(device)
            transfer_end = time.time()
            data_transfer_times.append(transfer_end - transfer_start)
            
            forward_start = time.time()
            outputs = model(rgb_inputs, morph_inputs)
            loss = criterion(outputs, targets)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)
            
            test_loss += loss.item() * rgb_inputs.size(0)
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
    
    test_loss /= len(test_loader.dataset)
    
    metrics_start = time.time()
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
    
    metrics_end = time.time()
    test_end = time.time()
    
    print(f"[TIME] Testing completed in {test_end - test_start:.2f}s")
    print(f"[TIME] Avg test batch: {np.mean(batch_times):.4f}s, "
          f"Avg test transfer: {np.mean(data_transfer_times):.4f}s, "
          f"Avg test forward: {np.mean(forward_times):.4f}s")
    print(f"[TIME] Metrics calculation: {metrics_end - metrics_start:.4f}s")
    
    return test_loss, results, all_outputs


def train_evaluation_pipeline(data_directory, biomarkers, morphology_features, num_epochs=5):
    overall_start = time.time()
    
    folders_start = time.time()
    folders = [os.path.join(data_directory, d) for d in os.listdir(data_directory) 
               if os.path.isdir(os.path.join(data_directory, d))]
    
    if not folders:
        print("No folders found in data directory")
        return
    
    single_folder = folders[0]
    folders_end = time.time()
    print(f"[TIME] Folder discovery: {folders_end - folders_start:.4f}s")
    print(f"Using folder: {single_folder}")
    
    features_start = time.time()
    print(f"Requested morphology features: {morphology_features}")
    
    csv_file = None
    he_path = None
    file_index = os.path.basename(single_folder)
    
    for file in os.listdir(single_folder):
        file_path = os.path.join(single_folder, file)
        if file.endswith('.csv') and file_index in file:
            csv_file = file_path
        elif 'registered.ome.tif' in file:
            he_path = file_path
    
    if csv_file is None or he_path is None:
        print(f"Missing files for {single_folder}")
        return
    
    df = pd.read_csv(csv_file)
    available_features = [f for f in morphology_features if f in df.columns]
    
    common_features = set(available_features)
    required_features = ["Area", "X_centroid", "Y_centroid"]
    for feature in required_features:
        if feature not in common_features:
            common_features.add(feature)
    
    common_morphology_features = sorted(list(common_features))
    print(f"Using morphology features: {common_morphology_features}")
    features_end = time.time()
    print(f"[TIME] Features discovery: {features_end - features_start:.4f}s")
    
    datasets_start = time.time()
    he_image = load_image(he_path)
    
    full_dataset = HEBiomarkerDataset(df, he_image, biomarkers, common_morphology_features)
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples")
    datasets_end = time.time()
    print(f"[TIME] Dataset loading: {datasets_end - datasets_start:.2f}s")
    
    loader_start = time.time()
    num_morphology_features = len(common_morphology_features)
    print(f"Using {num_morphology_features} morphology features")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    loader_end = time.time()
    print(f"[TIME] DataLoader creation: {loader_end - loader_start:.4f}s")
    
    model_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BiomarkerMLP(len(biomarkers), num_morphology_features).to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model has {model_size} parameters")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    model_end = time.time()
    print(f"[TIME] Model creation: {model_end - model_start:.4f}s")
    
    best_val_loss = float('inf')
    model_dir = os.path.join(data_directory, "model_results")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Training model for {num_epochs} epochs...")
    training_start = time.time()
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers,
                'morphology_features': common_morphology_features
            }, os.path.join(model_dir, "best_biomarker_model.pt"))
    
    training_end = time.time()
    print(f"[TIME] Total training: {training_end - training_start:.2f}s")
    
    testing_start = time.time()
    checkpoint = torch.load(os.path.join(model_dir, "best_biomarker_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, results, all_outputs = test_model(model, test_loader, criterion, device, biomarkers)
    testing_end = time.time()
    print(f"[TIME] Total testing: {testing_end - testing_start:.2f}s")
    
    output_start = time.time()
    
    results_df = pd.DataFrame(results).T
    
    output_dir = os.path.join(data_directory, "model_results_timed")
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics.csv"))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'biomarkers': biomarkers,
        'morphology_features': common_morphology_features
    }, os.path.join(output_dir, "biomarker_model_timed.pt"))
    
    output_end = time.time()
    print(f"[TIME] Output and saving: {output_end - output_start:.2f}s")
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    print("\n=== TIMING SUMMARY ===")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Data loading time: {datasets_end - datasets_start:.2f}s ({(datasets_end - datasets_start)/total_time*100:.1f}%)")
    print(f"Training time: {training_end - training_start:.2f}s ({(training_end - training_start)/total_time*100:.1f}%)")
    print(f"Testing time: {testing_end - testing_start:.2f}s ({(testing_end - testing_start)/total_time*100:.1f}%)")
    
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
    
    results = train_evaluation_pipeline(data_directory, biomarkers, morphology_features, num_epochs=5)