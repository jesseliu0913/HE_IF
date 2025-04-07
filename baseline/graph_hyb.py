import os
import tifffile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
import random
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.cuda.amp import GradScaler
import concurrent.futures
from joblib import Parallel, delayed
import time
import sys

class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.print_end = print_end
        self.start_time = time.time()
        self.iteration = 0
        self.print_progress()
    
    def print_progress(self):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        
        elapsed_time = time.time() - self.start_time
        if self.iteration > 0:
            estimated_total = elapsed_time * self.total / self.iteration
            eta = estimated_total - elapsed_time
            eta_str = f"ETA: {int(eta//60)}m {int(eta%60)}s"
        else:
            eta_str = "ETA: ?"
            
        sys.stdout.write(f'\r{self.prefix} |{bar}| {percent}% {self.suffix} {eta_str}{self.print_end}')
        sys.stdout.flush()
        
    def update(self, n=1):
        self.iteration += n
        self.print_progress()
    
    def close(self):
        sys.stdout.write('\n')
        

class EnhancedSuperpixelGraphDataset(Dataset):
    def __init__(self, he_image, df, biomarkers, patch_size=224, stride=224, transform=None, n_jobs=1):
        super(EnhancedSuperpixelGraphDataset, self).__init__()
        self.he_image = he_image
        self.df = df
        self.biomarkers = biomarkers
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, os.cpu_count() - 1)
        
        self.height, self.width = he_image.shape[:2]
        self.grid_h = (self.height - patch_size) // stride + 1
        self.grid_w = (self.width - patch_size) // stride + 1
        
        self.grid_positions = []
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                y = i * stride
                x = j * stride
                self.grid_positions.append((y, x, i, j))
        
        self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor.fc = nn.Identity()
        if torch.cuda.is_available():
            self.feature_extractor = self.feature_extractor.cuda()
        self.feature_extractor.eval()
        
        self._precompute_cell_assignments()
    
    def _extract_patch_features_batch(self, patches):
        with torch.cuda.amp.autocast():
            if self.transform:
                transformed_patches = torch.stack([self.transform(patch) for patch in patches])
            else:
                transformed_patches = torch.stack([
                    torch.tensor(np.transpose(patch, (2, 0, 1)), dtype=torch.float32)
                    for patch in patches
                ])
            
            if torch.cuda.is_available():
                transformed_patches = transformed_patches.cuda()
            
            with torch.no_grad():
                features = self.feature_extractor(transformed_patches)
            
            return features.cpu().numpy()
        
    def _precompute_cell_assignments(self):
        cell_to_patch = {}
        cell_indices = []
        
        patch_bounds = []
        for patch_idx, (patch_y, patch_x, _, _) in enumerate(self.grid_positions):
            patch_bounds.append((patch_idx, patch_y, patch_x, patch_y + self.patch_size, patch_x + self.patch_size))
        
        print(f"Assigning {len(self.df)} cells to patches...")
        progress = ProgressBar(len(self.df), prefix='Assigning Cells:', suffix='Complete', length=50)
        
        def process_cell_batch(cell_batch):
            local_cell_to_patch = {}
            local_cell_indices = []
            
            for cell_idx, (_, cell) in cell_batch:
                y, x = int(cell['Y_centroid']), int(cell['X_centroid'])
                
                potential_patches = []
                for patch_idx, min_y, min_x, max_y, max_x in patch_bounds:
                    if min_y <= y < max_y and min_x <= x < max_x:
                        potential_patches.append((patch_idx, min_y, min_x))
                
                if potential_patches:
                    if len(potential_patches) == 1:
                        best_patch = potential_patches[0][0]
                    else:
                        min_dist = float('inf')
                        best_patch = None
                        for patch_idx, patch_y, patch_x in potential_patches:
                            patch_center_y = patch_y + self.patch_size // 2
                            patch_center_x = patch_x + self.patch_size // 2
                            dist = (y - patch_center_y)**2 + (x - patch_center_x)**2
                            if dist < min_dist:
                                min_dist = dist
                                best_patch = patch_idx
                    
                    local_cell_to_patch[cell_idx] = best_patch
                    local_cell_indices.append(cell_idx)
            
            return local_cell_to_patch, local_cell_indices, len(cell_batch)
        
        cell_batches = []
        batch_size = max(500, len(self.df) // (self.n_jobs * 2))
        for i in range(0, len(self.df), batch_size):
            cell_batches.append(list(enumerate(self.df.iloc[i:i+batch_size].iterrows(), i)))
        
        results = []
        total_processed = 0
        
        for batch in cell_batches:
            batch_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_cell_batch)([batch[i]]) for i in range(len(batch))
            )
            
            for local_to_patch, local_indices, processed_count in batch_results:
                cell_to_patch.update(local_to_patch)
                cell_indices.extend(local_indices)
                total_processed += processed_count
                progress.update(processed_count)
        
        progress.close()
        
        train_indices, temp_indices = train_test_split(cell_indices, test_size=0.3, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
        
        self.cell_to_patch_mapping = {
            'cell_to_patch': cell_to_patch,
            'cell_indices': cell_indices
        }
        self.train_cell_indices = train_indices
        self.val_cell_indices = val_indices
        self.test_cell_indices = test_indices
    
    def create_node_features(self):
        node_features = np.zeros((len(self.grid_positions), 2048), dtype=np.float32)
        
        batch_size = 32
        total_batches = (len(self.grid_positions) + batch_size - 1) // batch_size
        print(f"Extracting features for {len(self.grid_positions)} patches...")
        
        progress = ProgressBar(len(self.grid_positions), prefix='Feature Extraction:', suffix='Complete', length=50)
        
        for i in range(0, len(self.grid_positions), batch_size):
            batch_indices = list(range(i, min(i+batch_size, len(self.grid_positions))))
            patches = []
            
            for idx in batch_indices:
                y, x, _, _ = self.grid_positions[idx]
                patch = self.he_image[y:y+self.patch_size, x:x+self.patch_size].astype(np.float32) / 255.0
                patches.append(patch)
            
            batch_features = self._extract_patch_features_batch(patches)
            for j, idx in enumerate(batch_indices):
                node_features[idx] = batch_features[j]
            
            progress.update(len(batch_indices))
        
        progress.close()
        return node_features
    
    def create_edge_index(self):
        print("Creating edge indices for graph connectivity...")
        edges = []
        
        node_positions = {}
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                node_idx = i * self.grid_w + j
                node_positions[(i, j)] = node_idx
        
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                node_idx = node_positions[(i, j)]
                
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_h and 0 <= nj < self.grid_w:
                            neighbor_idx = node_positions[(ni, nj)]
                            edges.append((node_idx, neighbor_idx))
                
                if (i + j) % 5 == 0:
                    for jump in [2, 3]:
                        if j + jump < self.grid_w:
                            neighbor_idx = node_positions[(i, j + jump)]
                            edges.append((node_idx, neighbor_idx))
                            edges.append((neighbor_idx, node_idx))
                        
                        if i + jump < self.grid_h:
                            neighbor_idx = node_positions[(i + jump, j)]
                            edges.append((node_idx, neighbor_idx))
                            edges.append((neighbor_idx, node_idx))
        
        print(f"Created {len(edges)} edges for the graph.")
        return torch.tensor(edges, dtype=torch.long).t()
    
    def create_cell_features_and_labels(self):
        total_cells = len(self.cell_to_patch_mapping['cell_indices'])
        
        morphology_features = [
            'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 
            'Solidity', 'Extent', 'Orientation'
        ]
        
        cell_features = np.zeros((total_cells, 9), dtype=np.float32)
        cell_patches = np.zeros(total_cells, dtype=np.int64)
        cell_labels = np.zeros((total_cells, len(self.biomarkers)), dtype=np.float32)
        
        print(f"Creating features for {total_cells} cells...")
        progress = ProgressBar(total_cells, prefix='Cell Features:', suffix='Complete', length=50)
        
        batch_size = 20000
        for batch_start in range(0, total_cells, batch_size):
            batch_end = min(batch_start + batch_size, total_cells)
            
            for new_idx in range(batch_start, batch_end):
                cell_idx = self.cell_to_patch_mapping['cell_indices'][new_idx]
                patch_idx = self.cell_to_patch_mapping['cell_to_patch'][cell_idx]
                cell_patches[new_idx] = patch_idx
                
                cell = self.df.iloc[cell_idx]
                cell_y, cell_x = int(cell['Y_centroid']), int(cell['X_centroid'])
                patch_y, patch_x, _, _ = self.grid_positions[patch_idx]
                
                rel_y = (cell_y - patch_y) / self.patch_size
                rel_x = (cell_x - patch_x) / self.patch_size
                
                feature_vector = [rel_y, rel_x]
                
                for feat in morphology_features:
                    if feat in cell:
                        value = cell[feat]
                        if feat == 'Area':
                            value = np.log1p(value) / 10
                        elif feat in ['MajorAxisLength', 'MinorAxisLength']:
                            value = value / 100
                        feature_vector.append(value)
                    else:
                        feature_vector.append(0.0)
                
                cell_features[new_idx, :len(feature_vector)] = feature_vector
                
                for i, biomarker in enumerate(self.biomarkers):
                    cell_labels[new_idx, i] = cell[biomarker]
            
            progress.update(batch_end - batch_start)
        
        progress.close()
        return cell_features, cell_patches, cell_labels
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        print("Building graph dataset...")
        node_features = self.create_node_features()
        edge_index = self.create_edge_index()
        cell_features, cell_patches, cell_labels = self.create_cell_features_and_labels()
        
        train_mask = np.zeros(len(cell_features), dtype=bool)
        val_mask = np.zeros(len(cell_features), dtype=bool)
        test_mask = np.zeros(len(cell_features), dtype=bool)
        
        cell_idx_mapping = {original_idx: new_idx for new_idx, original_idx in 
                           enumerate(self.cell_to_patch_mapping['cell_indices'])}
        
        print("Setting up train/validation/test masks...")
        for cell_idx in self.train_cell_indices:
            new_idx = cell_idx_mapping[cell_idx]
            train_mask[new_idx] = True
        
        for cell_idx in self.val_cell_indices:
            new_idx = cell_idx_mapping[cell_idx]
            val_mask[new_idx] = True
        
        for cell_idx in self.test_cell_indices:
            new_idx = cell_idx_mapping[cell_idx]
            test_mask[new_idx] = True
        
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            cell_features=torch.tensor(cell_features, dtype=torch.float),
            cell_patches=torch.tensor(cell_patches, dtype=torch.long),
            cell_labels=torch.tensor(cell_labels, dtype=torch.float),
            train_mask=torch.tensor(train_mask, dtype=torch.bool),
            val_mask=torch.tensor(val_mask, dtype=torch.bool),
            test_mask=torch.tensor(test_mask, dtype=torch.bool),
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            cell_idx_mapping=cell_idx_mapping
        )
        
        return data


class EnhancedCellLevelGNN(nn.Module):
    def __init__(self, patch_feature_dim, cell_feature_dim, hidden_dim, output_dim, num_layers=3, dropout_rate=0.3):
        super(EnhancedCellLevelGNN, self).__init__()
        
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_feature_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            else:
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False))
        
        self.cell_embedding = nn.Sequential(
            nn.Linear(cell_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.patch_embedding(x)
        
        for i, conv in enumerate(self.convs):
            if i == 0:
                x_res = x
            else:
                x_res = x
                
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
            
            if i > 0 and i % 2 == 0:
                x = x + x_res
        
        cell_features = data.cell_features
        cell_features = self.cell_embedding(cell_features)
        
        cell_patches = data.cell_patches
        cell_patch_features = x[cell_patches]
        
        combined = torch.cat([cell_patch_features, cell_features], dim=1)
        combined = self.combined_layer(combined)
        
        outputs = self.output_network(combined)
        
        return outputs


def train_enhanced_cell_gnn(data_directory, biomarkers, num_epochs=30, hidden_dim=512, 
                        patch_size=224, stride=224, log_transform=True, n_jobs=4):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU.")
    
    device = torch.device("cuda")
    scaler = GradScaler(enabled=True)
    
    folders = [os.path.join(data_directory, d) for d in os.listdir(data_directory) 
               if os.path.isdir(os.path.join(data_directory, d))]

    if not folders:
        raise ValueError("No valid folders found in the data directory")
    
    selected_folder = folders[0]
    print(f"Using folder: {selected_folder}")

    output_directory = "./results"
    os.makedirs(output_directory, exist_ok=True)
    
    file_index = os.path.basename(selected_folder)
    csv_file = None
    he_path = None
    
    for file in os.listdir(selected_folder):
        file_path = os.path.join(selected_folder, file)
        if file.endswith('.csv') and file_index in file:
            csv_file = file_path
        elif 'registered.ome.tif' in file:
            he_path = file_path
    
    if csv_file is None or he_path is None:
        raise ValueError(f"Missing required files in {selected_folder}")
    
    print(f"Processing {selected_folder}...")
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loading image file: {he_path}")
    he_image = tifffile.imread(he_path)
    print(f"Image shape: {he_image.shape}")
    
    if log_transform:
        print("Applying log(1+x) transform to biomarker values...")
        for biomarker in biomarkers:
            if (df[biomarker] < 0).any():
                min_val = df[biomarker].min()
                if min_val < 0:
                    print(f"Warning: Negative values found in {biomarker}, shifting by {abs(min_val)}")
                    df[biomarker] = df[biomarker] - min_val + 1e-6
            
            df[biomarker] = np.log1p(df[biomarker])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Creating enhanced superpixel graph dataset...")
    dataset = EnhancedSuperpixelGraphDataset(
        he_image=he_image,
        df=df,
        biomarkers=biomarkers, 
        patch_size=patch_size,
        stride=stride,
        transform=transform,
        n_jobs=n_jobs
    )
    
    print("Getting data from dataset...")
    data = dataset[0]
    data = data.to(device)
    
    train_size = data.train_mask.sum().item()
    val_size = data.val_mask.sum().item()
    test_size = data.test_mask.sum().item()
    
    print(f"Train cells: {train_size}")
    print(f"Validation cells: {val_size}")
    print(f"Test cells: {test_size}")
    
    patch_feature_dim = data.x.size(1)
    cell_feature_dim = data.cell_features.size(1)
    output_dim = len(biomarkers)
    
    print("Creating enhanced model...")
    model = EnhancedCellLevelGNN(
        patch_feature_dim=patch_feature_dim, 
        cell_feature_dim=cell_feature_dim, 
        hidden_dim=hidden_dim, 
        output_dim=output_dim,
        num_layers=3,
        dropout_rate=0.3
    )
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    criterion = nn.SmoothL1Loss()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    total_steps = num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.003, total_steps=total_steps, 
        pct_start=0.3, anneal_strategy='cos'
    )
    
    output_dir = os.path.join(output_directory, "model_results_enhanced_cell_gnn")
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    print(f"Training Enhanced Cell-Level GNN for {num_epochs} epochs...")
    epoch_progress = ProgressBar(num_epochs, prefix='Training Progress:', suffix='Epochs', length=50)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            all_outputs = model(data)
            
            train_outputs = all_outputs[data.train_mask]
            train_targets = data.cell_labels[data.train_mask]
            
            loss = criterion(train_outputs, train_targets)
        
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = all_outputs[data.val_mask]
            val_targets = data.cell_labels[data.val_mask]
            val_loss = criterion(val_outputs, val_targets).item()
            val_losses.append(val_loss)
        
        print(f"\rEpoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers,
            }, os.path.join(output_dir, "best_biomarker_cell_gnn.pt"))
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        epoch_progress.update(1)
    
    epoch_progress.close()
    
    print("Loading best model checkpoint...")
    checkpoint = torch.load(os.path.join(output_dir, "best_biomarker_cell_gnn.pt"))
    
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    with torch.no_grad():
        print("Generating predictions for test cells...")
        test_progress = ProgressBar(1, prefix='Generating Predictions:', suffix='Complete', length=50)
        
        all_outputs = model(data)
        test_outputs = all_outputs[data.test_mask].cpu().numpy()
        test_targets = data.cell_labels[data.test_mask].cpu().numpy()
        
        test_cell_indices = []
        for original_idx, new_idx in data.cell_idx_mapping.items():
            if data.test_mask[new_idx]:
                test_cell_indices.append(original_idx)
        
        test_df = df.iloc[test_cell_indices].copy()
        
        for i, biomarker in enumerate(biomarkers):
            if log_transform:
                test_df[f"{biomarker}_predicted"] = np.expm1(test_outputs[:, i])
                test_df[f"{biomarker}_abs_error"] = np.abs(np.expm1(test_targets[:, i]) - np.expm1(test_outputs[:, i]))
                test_df[f"{biomarker}_rel_error"] = test_df[f"{biomarker}_abs_error"] / (np.expm1(test_targets[:, i]) + 1e-6)
            else:
                test_df[f"{biomarker}_predicted"] = test_outputs[:, i]
                test_df[f"{biomarker}_abs_error"] = np.abs(test_targets[:, i] - test_outputs[:, i])
                test_df[f"{biomarker}_rel_error"] = test_df[f"{biomarker}_abs_error"] / (np.abs(test_targets[:, i]) + 1e-6)
        
        test_progress.update(1)
        test_progress.close()
        
        print("Saving predictions to CSV...")
        test_df.to_csv(os.path.join(output_dir, "cell_predictions.csv"), index=False)
        
        error_stats = {}
        for biomarker in biomarkers:
            error_stats[biomarker] = {
                'mean_abs_error': test_df[f"{biomarker}_abs_error"].mean(),
                'median_abs_error': test_df[f"{biomarker}_abs_error"].median(),
                'mean_rel_error': test_df[f"{biomarker}_rel_error"].mean(),
                'median_rel_error': test_df[f"{biomarker}_rel_error"].median(),
            }
        
        error_stats_df = pd.DataFrame(error_stats).T
        error_stats_df.to_csv(os.path.join(output_dir, "prediction_error_stats.csv"))
    
    results = {}
    
    print("Calculating evaluation metrics...")
    metric_progress = ProgressBar(len(biomarkers), prefix='Evaluation Metrics:', suffix='Complete', length=50)
    
    for i, biomarker in enumerate(biomarkers):
        pearson_r, _ = pearsonr(test_targets[:, i], test_outputs[:, i])
        spearman_r, _ = spearmanr(test_targets[:, i], test_outputs[:, i])
        
        if log_transform:
            actual_orig = np.expm1(test_targets[:, i])
            pred_orig = np.expm1(np.clip(test_outputs[:, i], -10, 10))
            c_index = concordance_index(actual_orig, pred_orig)
        else:
            c_index = concordance_index(test_targets[:, i], test_outputs[:, i])
        
        results[biomarker] = {
            'PearsonR': pearson_r,
            'SpearmanR': spearman_r,
            'C-index': c_index
        }
        
        metric_progress.update(1)
    
    metric_progress.close()
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics_cell_gnn.csv"))
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
        print("Loss curves visualization saved.")
    except:
        print("Could not save loss visualization.")
    
    print(f"Results saved to {output_dir}")
    print("Processing completed successfully!")
    return results_df

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    data_directory = "../data/data"
    biomarkers = [
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
        "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"
    ]
    
    patch_size = 224
    stride = 224
    
    torch.cuda.empty_cache()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.85)
    
    print("Starting training process...")
    train_enhanced_cell_gnn(
        data_directory=data_directory, 
        biomarkers=biomarkers, 
        num_epochs=30,
        hidden_dim=512,
        patch_size=patch_size,
        stride=stride,
        log_transform=True,
        n_jobs=4
    )