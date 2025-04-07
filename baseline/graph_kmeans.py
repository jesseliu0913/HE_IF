import os
import tifffile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import KDTree
from lifelines.utils import concordance_index
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.cuda.amp import GradScaler
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import exposure


def extract_features(patch, feature_extractor=None):
    if feature_extractor is not None:
        patch_tensor = torch.tensor(np.transpose(patch, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
        patch_tensor = patch_tensor.to(next(feature_extractor.parameters()).device)
        with torch.no_grad():
            features = feature_extractor(patch_tensor)
        return features.squeeze().cpu().numpy()
    else:
        patch_gray = rgb2gray(patch)
        features, _ = hog(
            patch_gray, 
            orientations=8, 
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2), 
            visualize=True, 
            feature_vector=True
        )
        return features

def construct_cell_graph(df, he_image, biomarkers, feature_extractor=None, k_neighbors=5):
    centroids = df[['X_centroid', 'Y_centroid']].values
    tree = KDTree(centroids)
    _, indices = tree.query(centroids, k=k_neighbors+1)  
    
    patch_size = 224
    half_size = patch_size // 2
    
    node_features = []
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = int(row["X_centroid"]), int(row["Y_centroid"])
        
        x_min, x_max = max(0, x - half_size), min(he_image.shape[1], x + half_size)
        y_min, y_max = max(0, y - half_size), min(he_image.shape[0], y + half_size)
        
        if x_max <= x_min or y_max <= y_min or x_min >= he_image.shape[1] or y_min >= he_image.shape[0]:
            patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        else:
            patch_height = y_max - y_min
            patch_width = x_max - x_min
            
            patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
            
            extract = he_image[y_min:y_max, x_min:x_max]
            
            y_offset = (patch_size - patch_height) // 2
            x_offset = (patch_size - patch_width) // 2
            
            patch[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width] = extract
        
        patch = patch.astype(np.float32) / 255.0
        
        features = extract_features(patch, feature_extractor)
        node_features.append(features)
    
    edge_indices = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors[1:]:  
            edge_indices.append((i, neighbor))
            edge_indices.append((neighbor, i)) 
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    y = torch.zeros((len(df), len(biomarkers)), dtype=torch.float)
    for i, biomarker in enumerate(biomarkers):
        y[:, i] = torch.tensor(np.log1p(df[biomarker].values), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data, df.index.values


class CellGraphDataset(Dataset):
    def __init__(self, graph_data, cell_indices, labels):
        super(CellGraphDataset, self).__init__()
        self.graph_data = graph_data
        self.cell_indices = cell_indices
        self.labels = labels
    
    def len(self):
        return len(self.cell_indices)
    
    def get(self, idx):
        cell_idx = self.cell_indices[idx]
        neighbors = self.graph_data.edge_index[1][self.graph_data.edge_index[0] == cell_idx].unique()
        nodes = torch.cat([torch.tensor([cell_idx]), neighbors])
        
        x = self.graph_data.x[nodes]
        y = self.labels[cell_idx]
        
        node_mapping = {int(nodes[i]): i for i in range(len(nodes))}
        edge_index = []
        for e in range(self.graph_data.edge_index.size(1)):
            src, dst = int(self.graph_data.edge_index[0, e]), int(self.graph_data.edge_index[1, e])
            if src in node_mapping and dst in node_mapping:
                edge_index.append([node_mapping[src], node_mapping[dst]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        central_node_idx = torch.tensor([0], dtype=torch.long) 
        
        return Data(x=x, edge_index=edge_index, y=y, central_node_idx=central_node_idx)


class GNNBiomarkerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GNNBiomarkerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, data):
        x, edge_index, batch, central_node_idx = data.x, data.edge_index, data.batch, data.central_node_idx
        x = self.embedding(x)
        x = F.relu(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        
        central_embeddings = x[central_node_idx]
        
        x = F.relu(self.lin1(central_embeddings))
        x = self.lin2(x)
        
        return x

def train_gnn_pipeline(data_directory, biomarkers, batch_size=128, num_epochs=30, hidden_dim=256):
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
    df = pd.read_csv(csv_file)
    he_image = tifffile.imread(he_path)
    
    indices = np.arange(len(df))
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_indices)} cells")
    print(f"Validation set: {len(val_indices)} cells")
    print(f"Test set: {len(test_indices)} cells")

    feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor.fc = nn.Identity()  
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    
    print("Constructing cell graph from H&E image...")
    graph_data, all_indices = construct_cell_graph(df, he_image, biomarkers, feature_extractor)
    
    train_dataset = CellGraphDataset(graph_data, train_indices, graph_data.y)
    val_dataset = CellGraphDataset(graph_data, val_indices, graph_data.y)
    test_dataset = CellGraphDataset(graph_data, test_indices, graph_data.y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, 
                             pin_memory=True, follow_batch=['x'], persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, 
                           pin_memory=True, follow_batch=['x'], persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, 
                            pin_memory=True, follow_batch=['x'], persistent_workers=True)
    
    input_dim = graph_data.x.size(1)
    output_dim = len(biomarkers)
    
    model = GNNBiomarkerModel(input_dim, hidden_dim, output_dim)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )
    
    output_dir = os.path.join(output_directory, "model_results_gnn")
    os.makedirs(output_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print(f"Training GNN for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}"):
                batch = batch.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(batch)
                    loss = criterion(outputs, batch.y)
                
                val_loss += loss.item() * batch.num_graphs
        
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers
            }, os.path.join(output_dir, "best_biomarker_gnn.pt"))
    
    torch.cuda.empty_cache()
    
    checkpoint = torch.load(os.path.join(output_dir, "best_biomarker_gnn.pt"))
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                
            test_loss += loss.item() * batch.num_graphs
            
            all_targets.append(batch.y.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            all_indices.extend(test_indices[batch.batch.cpu().numpy() == 0])
    
    torch.cuda.empty_cache()
    
    test_loss /= len(test_dataset)
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
    
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics_gnn.csv"))
    
    test_df = df.iloc[test_indices].copy()
    for i, biomarker in enumerate(biomarkers):
        test_df[f"{biomarker}_predicted"] = np.expm1(np.clip(all_outputs[:, i], -10, 10))
    
    test_df.to_csv(os.path.join(output_dir, f"cell_predictions_gnn.csv"), index=False)
    
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save({
        'model_state_dict': model_state_dict,
        'biomarkers': biomarkers,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim
    }, os.path.join(output_dir, "biomarker_model_gnn.pt"))
    
    print(f"Results saved to {output_dir}")
    return results_df

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    data_directory = "../data/data"
    biomarkers = [
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
        "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"
    ]
    
    batch_size = 64  
    torch.cuda.empty_cache()
    
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    train_gnn_pipeline(data_directory, biomarkers, batch_size=batch_size, num_epochs=5)