import numpy as np
import pandas as pd
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


os.environ["OPENBLAS_NUM_THREADS"] = "16"  # Limit OpenBLAS threads
os.environ["MKL_NUM_THREADS"] = "16"       # Limit MKL threads
os.environ["OMP_NUM_THREADS"] = "16"       # Limit OpenMP threads
os.environ["NUMEXPR_NUM_THREADS"] = "16"   # Limit numexpr threads
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # Limit vecLib threads


def load_cell_features(data_path="/playpen/jesse/HE_IF/graph_comparison/cell_feature"):
    feature_data = np.load(f"{data_path}/cell_features.npz")
    cell_ids = feature_data['cell_ids']
    features = feature_data['features']
    coords = feature_data['coords']
    
    with open(f"{data_path}/data_splits.json", 'r') as f:
        data_splits = json.load(f)
    
    return cell_ids, features, coords, data_splits

def construct_knn_graph(features, coords, k=20, use_coords=False, weighted=True):
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    
    data = coords if use_coords else features

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data)
    
    distances, indices = nbrs.kneighbors(data)
    
    n_cells = data.shape[0]
    rows = np.repeat(np.arange(n_cells), k)
    cols = indices[:, 1:].flatten()  
    
    if weighted:
        sigma = np.mean(distances[:, 1:])  
        weights = np.exp(-distances[:, 1:] ** 2 / (2 * sigma ** 2)).flatten()
    else:
        weights = np.ones(rows.shape[0])
    
    adjacency = csr_matrix((weights, (rows, cols)), shape=(n_cells, n_cells))
    adjacency = adjacency.maximum(adjacency.transpose())
    
    edges = np.array(adjacency.nonzero()).T
    edge_weights = adjacency[edges[:, 0], edges[:, 1]]
    
    return edges, edge_weights, adjacency

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.out(x)
        
        return x

def prepare_pytorch_geometric_data(features, edges, edge_weights, targets, indices=None):
    if indices is not None:
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
        
        mask = np.isin(edges[:, 0], indices) & np.isin(edges[:, 1], indices)
        subgraph_edges = edges[mask]
        subgraph_weights = edge_weights[mask]
        
        remapped_edges = np.array([[idx_map[e[0]], idx_map[e[1]]] for e in subgraph_edges])
    else:
        remapped_edges = edges
        subgraph_weights = edge_weights
    
    x = torch.FloatTensor(features)
    edge_index = torch.LongTensor(remapped_edges.T)
    edge_attr = torch.FloatTensor(subgraph_weights)
    y = torch.FloatTensor(targets)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data

def train_model(model, train_loader, val_loader, device, num_epochs=20, lr=0.001, use_wandb=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model = None
    
    if use_wandb:
        import wandb
        wandb.init(project="biomarker-gnn", name="gnn-training-run")
        wandb.config.update({
            "learning_rate": lr,
            "epochs": num_epochs,
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__
        })
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
        
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
    
    model.load_state_dict(best_model)
    
    if use_wandb:
        wandb.finish()
    
    return model, train_losses, val_losses

def evaluate_model(model, loader, device, biomarkers):
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr)
            
            all_targets.append(data.y.cpu().numpy())
            all_outputs.append(out.cpu().numpy())
    
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
    
    return results, all_targets, all_outputs

def main():
    data_path = "/playpen/jesse/HE_IF/graph_comparison/cell_feature"
    cell_ids, features, coords, data_splits = load_cell_features(data_path)
    
    print(f"Loaded {len(cell_ids)} cells with {features.shape[1]} features each")
    
    edges, edge_weights, adj_matrix = construct_knn_graph(features, coords, k=20, use_coords=False)
    # spatial_edges, spatial_weights, spatial_adj = construct_knn_graph(features, coords, k=10, use_coords=True)
    
    csv_file = "/playpen/jesse/HIPI/preprocess/data/CRC03_new_coordinates.csv"
    df = pd.read_csv(csv_file)
    
    biomarker_cols = [
        'Hoechst1', 'Hoechst2', 'Hoechst3', 'Hoechst4', 'Hoechst5', 
        'Hoechst6', 'Hoechst7', 'Hoechst8', 'Hoechst9', 'A488', 
        'CD3', 'Ki67', 'CD4', 'CD20', 'CD163', 'Ecadherin', 
        'LaminABC', 'PCNA', 'A555', 'NaKATPase', 'Keratin', 
        'CD45', 'CD68', 'FOXP3', 'Vimentin', 'Desmin', 
        'Ki67_570', 'A647', 'CD45RO', 'aSMA', 'PD1', 
        'CD8a', 'PDL1', 'CDX2', 'CD31', 'Collagen'
    ]
    
    cell_id_to_biomarkers = {}
    for i, row in df.iterrows():
        cell_id = row.name
        cell_id_to_biomarkers[cell_id] = np.log1p(row[biomarker_cols].values)
    
    train_indices = data_splits['train']
    val_indices = data_splits['val']
    test_indices = data_splits['test']
    
    train_features = features[train_indices]
    val_features = features[val_indices]
    test_features = features[test_indices]
    
    biomarker_data = np.log1p(df[biomarker_cols].values)
    
    train_targets = biomarker_data[train_indices]
    val_targets = biomarker_data[val_indices]
    test_targets = biomarker_data[test_indices]
    
    print(f"Train set: {len(train_indices)} cells")
    print(f"Val set: {len(val_indices)} cells")
    print(f"Test set: {len(test_indices)} cells")
    
    scaler = StandardScaler()
    train_filtered_features = scaler.fit_transform(train_filtered_features)
    val_filtered_features = scaler.transform(val_filtered_features)
    test_filtered_features = scaler.transform(test_filtered_features)
    
    train_data = prepare_pytorch_geometric_data(
        train_filtered_features, edges, edge_weights, train_targets, train_filtered_indices)
    val_data = prepare_pytorch_geometric_data(
        val_filtered_features, edges, edge_weights, val_targets, val_filtered_indices)
    test_data = prepare_pytorch_geometric_data(
        test_filtered_features, edges, edge_weights, test_targets, test_filtered_indices)
    
    train_loader = DataLoader([train_data], batch_size=1)
    val_loader = DataLoader([val_data], batch_size=1)
    test_loader = DataLoader([test_data], batch_size=1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = train_filtered_features.shape[1]
    hidden_dim = 128
    output_dim = len(biomarker_cols)
    
    model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
    
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, num_epochs=20, use_wandb=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.close()
    
    results, all_targets, all_outputs = evaluate_model(model, test_loader, device, biomarker_cols)
    
    print("\nBiomarker Prediction Results:")
    for biomarker, metrics in results.items():
        print(f"{biomarker}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    avg_pearson = np.mean([metrics['PearsonR'] for metrics in results.values()])
    avg_spearman = np.mean([metrics['SpearmanR'] for metrics in results.values()])
    avg_cindex = np.mean([metrics['C-index'] for metrics in results.values()])
    
    print("\nOverall Performance:")
    print(f"Average Pearson R: {avg_pearson:.4f}")
    print(f"Average Spearman R: {avg_spearman:.4f}")
    print(f"Average C-index: {avg_cindex:.4f}")
    
    torch.save(model.state_dict(), "biomarker_gnn_model.pt")
    np.savez("prediction_results.npz", 
             targets=all_targets, 
             predictions=all_outputs, 
             biomarkers=biomarker_cols)
    
    return model, results

if __name__ == "__main__":
    main()