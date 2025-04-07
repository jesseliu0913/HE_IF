import os
import time
import tifffile
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights
import random


class CellPatchDataset(Dataset):
    def __init__(self, df, he_image, biomarkers):
        start_time = time.time()
        
        self.df = df
        self.he_image = he_image
        self.biomarkers = biomarkers
        
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
            patch = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            patch = self.he_image[y_min:y_max, x_min:x_max]
            
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                patch = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                from PIL import Image
                patch_pil = Image.fromarray(patch)
                patch_pil = patch_pil.resize((224, 224), Image.BILINEAR)
                patch = np.array(patch_pil)
        
        coordinates = np.array([x, y], dtype=np.float32)
        
        targets = np.zeros(len(self.biomarkers), dtype=np.float32)
        for i, biomarker in enumerate(self.biomarkers):
            targets[i] = np.log1p(row[biomarker]) if biomarker in row else 0.0
        
        return {
            'patch': patch,
            'coordinates': coordinates,
            'targets': targets,
            'cell_id': idx
        }


def collate_cell_graphs(batch):
    graph_construction_start = time.time()
    
    patches = [item['patch'] for item in batch]
    coordinates = [item['coordinates'] for item in batch]
    targets = [item['targets'] for item in batch]
    cell_ids = [item['cell_id'] for item in batch]
    
    coordinates = torch.tensor(np.stack(coordinates), dtype=torch.float)
    targets = torch.tensor(np.stack(targets), dtype=torch.float)
    cell_ids = torch.tensor(cell_ids, dtype=torch.long)
    
    num_nodes = len(batch)
    
    x_diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
    dist_matrix = torch.norm(x_diff, dim=2)
    
    threshold = dist_matrix.median() * 2
    adj_matrix = (dist_matrix < threshold).float()
    
    adj_matrix.fill_diagonal_(0)
    
    edge_index, _ = dense_to_sparse(adj_matrix)
    
    node_features = []
    for patch in patches:
        patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1) / 255.0
        node_features.append(patch_tensor)
    node_features = torch.stack(node_features)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=targets,
        cell_ids=cell_ids,
        batch=torch.zeros(num_nodes, dtype=torch.long)
    )
    
    graph_construction_end = time.time()
    if random.random() < 0.05:
        print(f"[TIME] Graph construction: {graph_construction_end - graph_construction_start:.4f}s for {num_nodes} nodes")
    
    return data


class ViTGNN(nn.Module):
    def __init__(self, num_biomarkers, hidden_dim=128, pretrained=True):
        super(ViTGNN, self).__init__()
        
        if pretrained:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.backbone = vit_b_16()
        
        self.backbone.heads = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # ViT hidden dimension is 768 for vit_b_16
        vit_feature_dim = 768
        
        self.conv1 = GCNConv(vit_feature_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_biomarkers)
        )
        
    def forward(self, data):
        vit_start = time.time()
        
        batch_size = data.x.size(0)
        
        with torch.no_grad():
            features = self.backbone(data.x)
            
        vit_end = time.time()
        
        gnn_start = time.time()
        # Apply GNN layers
        x = F.relu(self.conv1(features, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x, data.edge_index)
        
        # Apply MLP for prediction
        predictions = self.mlp(x)
        gnn_end = time.time()
        
        if random.random() < 0.05:
            print(f"[TIME] ViT feature extraction: {vit_end - vit_start:.4f}s, GNN processing: {gnn_end - gnn_start:.4f}s")
        
        return predictions


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
    forward_times = []
    backward_times = []
    total_samples = 0
    
    epoch_start = time.time()
    
    for i, data in enumerate(train_loader):
        batch_start = time.time()
        
        data_transfer_start = time.time()
        data = data.to(device)
        data_transfer_end = time.time()
        
        optimizer.zero_grad()
        
        forward_start = time.time()
        outputs = model(data)
        loss = criterion(outputs, data.y)
        forward_end = time.time()
        forward_time = forward_end - forward_start
        forward_times.append(forward_time)
        
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        backward_end = time.time()
        backward_time = backward_end - backward_start
        backward_times.append(backward_time)
        
        # Count samples in this batch
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
        total_samples += batch_size
        train_loss += loss.item() * batch_size
        
        batch_end = time.time()
        batch_time = batch_end - batch_start
        batch_times.append(batch_time)
        
        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(train_loader)}: total={batch_time:.4f}s, "
                  f"transfer={data_transfer_end - data_transfer_start:.4f}s, "
                  f"forward={forward_time:.4f}s, backward={backward_time:.4f}s")
    
    train_loss /= total_samples
    epoch_end = time.time()
    
    print(f"[TIME] Epoch {epoch_num} completed in {epoch_end - epoch_start:.2f}s")
    print(f"[TIME] Avg batch: {np.mean(batch_times):.4f}s, "
          f"Avg forward: {np.mean(forward_times):.4f}s, "
          f"Avg backward: {np.mean(backward_times):.4f}s")
    
    return train_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    batch_times = []
    total_samples = 0
    
    validate_start = time.time()
    
    with torch.no_grad():
        for data in val_loader:
            batch_start = time.time()
            
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data.y)
            
            # Count samples in this batch
            batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
            total_samples += batch_size
            val_loss += loss.item() * batch_size
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
    
    val_loss /= total_samples
    validate_end = time.time()
    
    print(f"[TIME] Validation completed in {validate_end - validate_start:.2f}s")
    print(f"[TIME] Avg val batch: {np.mean(batch_times):.4f}s")
    
    return val_loss


def test_model(model, test_loader, criterion, device, biomarkers):
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    batch_times = []
    total_samples = 0
    
    test_start = time.time()
    
    with torch.no_grad():
        for data in test_loader:
            batch_start = time.time()
            
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data.y)
            
            # Count samples in this batch
            batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
            total_samples += batch_size
            test_loss += loss.item() * batch_size
            
            all_targets.append(data.y.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
    
    test_loss /= total_samples
    
    metrics_start = time.time()
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    
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
    print(f"[TIME] Avg test batch: {np.mean(batch_times):.4f}s")
    print(f"[TIME] Metrics calculation: {metrics_end - metrics_start:.4f}s")
    
    return test_loss, results, all_outputs


def train_evaluation_pipeline(data_directory, biomarkers, num_epochs=5):
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
    
    file_search_start = time.time()
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
    file_search_end = time.time()
    print(f"[TIME] File search: {file_search_end - file_search_start:.4f}s")
    
    data_loading_start = time.time()
    df = pd.read_csv(csv_file)
    he_image = load_image(he_path)
    data_loading_end = time.time()
    print(f"[TIME] Data loading: {data_loading_end - data_loading_start:.2f}s")
    
    dataset_creation_start = time.time()
    full_dataset = CellPatchDataset(df, he_image, biomarkers)
    
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test samples")
    dataset_creation_end = time.time()
    print(f"[TIME] Dataset creation: {dataset_creation_end - dataset_creation_start:.2f}s")
    
    dataloader_start = time.time()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_cell_graphs)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_cell_graphs)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_cell_graphs)
    dataloader_end = time.time()
    print(f"[TIME] DataLoader creation: {dataloader_end - dataloader_start:.4f}s")
    
    model_init_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ViTGNN(len(biomarkers)).to(device)
    
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {model_size} trainable parameters")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    model_init_end = time.time()
    print(f"[TIME] Model initialization: {model_init_end - model_init_start:.4f}s")
    
    best_val_loss = float('inf')
    model_dir = os.path.join(data_directory, "vit_gnn_model_results")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Training model for {num_epochs} epochs...")
    training_start = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        epoch_end = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_end - epoch_start:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers
            }, os.path.join(model_dir, "best_biomarker_gnn_model.pt"))
    
    training_end = time.time()
    print(f"[TIME] Total training: {training_end - training_start:.2f}s")
    
    testing_start = time.time()
    checkpoint = torch.load(os.path.join(model_dir, "best_biomarker_gnn_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, results, all_outputs = test_model(model, test_loader, criterion, device, biomarkers)
    testing_end = time.time()
    print(f"[TIME] Total testing: {testing_end - testing_start:.2f}s")
    
    output_start = time.time()
    results_df = pd.DataFrame(results).T
    
    output_dir = os.path.join(data_directory, "vit_gnn_results")
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics.csv"))
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'biomarkers': biomarkers
    }, os.path.join(output_dir, "biomarker_vit_gnn_model.pt"))
    
    output_end = time.time()
    print(f"[TIME] Output and saving: {output_end - output_start:.2f}s")
    
    overall_end = time.time()
    total_time = overall_end - overall_start
    print("\n=== TIMING SUMMARY ===")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Data loading time: {data_loading_end - data_loading_start:.2f}s ({(data_loading_end - data_loading_start)/total_time*100:.1f}%)")
    print(f"Dataset creation: {dataset_creation_end - dataset_creation_start:.2f}s ({(dataset_creation_end - dataset_creation_start)/total_time*100:.1f}%)")
    print(f"Training time: {training_end - training_start:.2f}s ({(training_end - training_start)/total_time*100:.1f}%)")
    print(f"Testing time: {testing_end - testing_start:.2f}s ({(testing_end - testing_start)/total_time*100:.1f}%)")
    
    return results_df


if __name__ == "__main__":
    data_directory = "../data/data"
    biomarkers = [
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
        "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"
    ]
    
    results = train_evaluation_pipeline(data_directory, biomarkers, num_epochs=5)