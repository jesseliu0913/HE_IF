import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

def load_cell_features(data_path="/playpen/jesse/HE_IF/graph_comparison/cell_feature"):
    feature_data = np.load(f"{data_path}/cell_features.npz")
    cell_ids = feature_data['cell_ids']
    features = feature_data['features']
    coords = feature_data['coords']
    
    with open(f"{data_path}/data_splits.json", 'r') as f:
        data_splits = json.load(f)
    
    return cell_ids, features, coords, data_splits

def create_kmeans_graph(cell_features, cell_coords, n_clusters=None, k_neighbors=10, max_distance=50, 
                        batch_size=5000, memory_efficient=True, cache_path=None, cache_id=None):
    """
    Create K-means based graph with caching capability
    """
    # If cache path and ID are provided, try to load from cache
    if cache_path and cache_id:
        cache_file = f"{cache_path}/kmeans_graph_cache_{cache_id}.pkl"
        if os.path.exists(cache_file):
            print(f"Loading cached graph from {cache_file}...")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                G = cached_data["G"]
                data = cached_data["data"]
                return G, data
    
    print(f"Creating K-means graph for {len(cell_coords)} cells...")
    
    # Determine number of clusters if not specified
    if n_clusters is None:
        # A rule of thumb: sqrt of number of samples
        n_clusters = min(int(np.sqrt(len(cell_coords))), 100)
    
    print(f"Clustering into {n_clusters} clusters...")
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cell_coords)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in tqdm(range(len(cell_coords)), desc="Adding nodes"):
        G.add_node(i, pos=cell_coords[i], cluster=cluster_labels[i])
    
    # Connect nodes within the same cluster based on nearest neighbors
    print("Connecting nodes within clusters...")
    
    # Process each cluster separately to save memory
    for cluster_id in tqdm(range(n_clusters), desc="Processing clusters"):
        # Get indices of cells in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 1:  # Need at least 2 points for nearest neighbors
            # Get coordinates of cells in this cluster
            cluster_coords = cell_coords[cluster_indices]
            
            # Find k nearest neighbors for each cell in the cluster
            k = min(k_neighbors, len(cluster_indices) - 1)  # Can't have more neighbors than points - 1
            nn = NearestNeighbors(n_neighbors=k+1)  # +1 because the point itself is included
            nn.fit(cluster_coords)
            distances, indices = nn.kneighbors(cluster_coords)
            
            # Add edges based on nearest neighbors
            for i in range(len(cluster_indices)):
                source = cluster_indices[i]
                
                # Skip the first neighbor as it's the point itself
                for j, idx in enumerate(indices[i][1:], 1):
                    target = cluster_indices[idx]
                    dist = distances[i][j]
                    
                    # Only add edge if within max_distance
                    if max_distance is None or dist <= max_distance:
                        G.add_edge(int(source), int(target))
    
    # Connect neighboring clusters
    print("Connecting neighboring clusters...")
    
    # Find centroids of each cluster
    centroids = kmeans.cluster_centers_
    
    # Find nearest neighboring clusters for each cluster
    nn_clusters = NearestNeighbors(n_neighbors=3)  # Connect to 2 nearest clusters
    nn_clusters.fit(centroids)
    c_distances, c_indices = nn_clusters.kneighbors(centroids)
    
    # For each pair of neighboring clusters, connect their boundary points
    for i in range(n_clusters):
        cluster_i_indices = np.where(cluster_labels == i)[0]
        
        # Skip the first neighbor as it's the cluster itself
        for j in c_indices[i][1:]:
            cluster_j_indices = np.where(cluster_labels == j)[0]
            
            # Find the closest pair of points between the two clusters
            min_dist = float('inf')
            closest_pair = None
            
            # Use a sampling approach to reduce computational load for large clusters
            sample_size_i = min(len(cluster_i_indices), 100)
            sample_size_j = min(len(cluster_j_indices), 100)
            
            sampled_i = np.random.choice(cluster_i_indices, sample_size_i, replace=False)
            sampled_j = np.random.choice(cluster_j_indices, sample_size_j, replace=False)
            
            for idx_i in sampled_i:
                for idx_j in sampled_j:
                    dist = np.linalg.norm(cell_coords[idx_i] - cell_coords[idx_j])
                    if dist < min_dist and dist <= max_distance:
                        min_dist = dist
                        closest_pair = (idx_i, idx_j)
            
            # Connect the closest pair if found and within max_distance
            if closest_pair is not None:
                G.add_edge(int(closest_pair[0]), int(closest_pair[1]))
    
    print(f"Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Create edge_index for PyTorch Geometric
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create feature tensor
    if memory_efficient:
        x = torch.tensor(cell_features, dtype=torch.float32)
    else:
        x = torch.tensor(np.array([cell_features[i] for i in G.nodes()]), dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index)
    
    # Save to cache if cache path and ID are provided
    if cache_path and cache_id:
        os.makedirs(cache_path, exist_ok=True)
        cache_file = f"{cache_path}/kmeans_graph_cache_{cache_id}.pkl"
        print(f"Saving graph to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump({"G": G, "data": data}, f)
    
    return G, data

class BiomarkerGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=4, dropout=0.3, 
                 residual=True, edge_features=False):
        super(BiomarkerGNN, self).__init__()
        
        self.num_layers = num_layers
        self.residual = residual
        self.dropout = dropout
        self.edge_features = edge_features
        
        self.input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.input_norm = torch.nn.BatchNorm1d(hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        self.norms1 = torch.nn.ModuleList()
        self.norms2 = torch.nn.ModuleList()
        self.ffns = torch.nn.ModuleList()
        self.skip_connections = torch.nn.ModuleList()
        
        for i in range(num_layers):
            if i % 3 == 0:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif i % 3 == 1:
                self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=False))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            
            self.norms1.append(torch.nn.BatchNorm1d(hidden_dim))
            self.norms2.append(torch.nn.BatchNorm1d(hidden_dim))
            
            self.ffns.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim * 4),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim * 4, hidden_dim)
            ))
                
            if residual:
                self.skip_connections.append(torch.nn.Linear(hidden_dim, hidden_dim))
        
        self.final_norm = torch.nn.BatchNorm1d(hidden_dim)
        
        self.pre_output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout/2)
        )
        
        self.predictor = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for i in range(self.num_layers):
            identity = x
            
            try:
                x_conv = self.convs[i](x, edge_index)
                
                if self.residual:
                    x = x_conv + (identity if i == 0 else self.skip_connections[i](identity))
                else:
                    x = x_conv
                    
                x = self.norms1[i](x)
                
                ffn_out = self.ffns[i](x)
                x = x + ffn_out
                x = self.norms2[i](x)
                
                if i < self.num_layers - 1:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            except Exception as e:
                print(f"Error in layer {i} ({type(self.convs[i]).__name__}): {str(e)}")
                if self.residual:
                    x = identity
        
        x = self.final_norm(x)
        x = self.pre_output(x)
        x = self.predictor(x)
        
        return x

def train_model(model, train_data, val_data, train_targets, val_targets, num_epochs=100, patience=15, 
               learning_rate=1e-3, weight_decay=1e-5, output_dir="results", batch_size=32):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_data = val_data.to(device)
    val_targets = val_targets.to(device)
    model = model.to(device)
    
    # Create subgraph batches manually
    def create_subgraph_batch(data, indices):
        # Extract a subgraph containing only the nodes in indices
        node_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
        node_mask[indices] = True
        
        # Get edges where both nodes are in our batch
        edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
        batch_edge_index = data.edge_index[:, edge_mask]
        
        # Remap node indices to be consecutive
        node_idx = torch.zeros(data.x.size(0), dtype=torch.long)
        node_idx[indices] = torch.arange(len(indices))
        batch_edge_index = node_idx[batch_edge_index]
        
        return data.x[indices], batch_edge_index
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=5, verbose=True
    )
    criterion = nn.HuberLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    counter = 0
    
    num_nodes = train_data.x.size(0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle indices for this epoch
        indices = torch.randperm(num_nodes)
        
        for i in range(0, num_nodes, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_nodes)]
            
            # Create a subgraph for this batch
            x_batch, edge_index_batch = create_subgraph_batch(train_data, batch_indices)
            
            # Move batch to device
            x_batch = x_batch.to(device)
            edge_index_batch = edge_index_batch.to(device)
            y_batch = train_targets[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            try:
                out = model(x_batch, edge_index_batch)
                loss = criterion(out, y_batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * batch_indices.size(0)
                num_batches += 1
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {str(e)}")
                continue
        
        if num_batches > 0:
            train_loss = total_loss / (num_batches * batch_size)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                out = model(val_data.x, val_data.edge_index)
                val_loss = criterion(out, val_targets).item()
                val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict().copy()
                counter = 0
                
                torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    if best_model is not None:
        model.load_state_dict(best_model)
    return model, train_losses, val_losses

def evaluate_model(model, test_data, biomarkers, test_targets, output_dir="results", batch_size=512):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    model.eval()
    
    # Process the test data in batches
    num_nodes = test_data.x.size(0)
    all_outputs = []
    
    with torch.no_grad():
        for i in range(0, num_nodes, batch_size):
            end_idx = min(i + batch_size, num_nodes)
            batch_indices = torch.arange(i, end_idx)
            
            # For evaluation, we can use the whole graph structure but just predict 
            # for a subset of nodes to save memory
            x_batch = test_data.x.to(device)
            edge_index_batch = test_data.edge_index.to(device)
            
            # Only compute outputs for the current batch of nodes
            batch_out = model(x_batch, edge_index_batch)[i:end_idx]
            all_outputs.append(batch_out.cpu())
    
    # Combine all batch outputs
    outputs = torch.cat(all_outputs, dim=0).numpy()
    targets = test_targets.numpy()
    
    results = {}
    pearson_values = []
    spearman_values = []
    cindex_values = []

    for i, biomarker in enumerate(biomarkers):
        pearson_r, p_value = pearsonr(targets[:, i], outputs[:, i])
        spearman_r, _ = spearmanr(targets[:, i], outputs[:, i])
        
        actual_orig = np.expm1(targets[:, i])
        pred_orig = np.expm1(np.clip(outputs[:, i], -10, 10))
        c_index = concordance_index(actual_orig, pred_orig)
        
        results[biomarker] = {
            'PearsonR': pearson_r,
            'PearsonP': p_value,
            'SpearmanR': spearman_r,
            'C-index': c_index
        }
        
        pearson_values.append(pearson_r)
        spearman_values.append(spearman_r)
        cindex_values.append(c_index)
        
        print(f"{biomarker}: Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}, C-index={c_index:.4f}")
    
    avg_pearson = np.mean(pearson_values)
    avg_spearman = np.mean(spearman_values)
    avg_cindex = np.mean(cindex_values)
    
    print(f"\nAVERAGE METRICS:")
    print(f"Average Pearson R: {avg_pearson:.4f}")
    print(f"Average Spearman R: {avg_spearman:.4f}")
    print(f"Average C-index: {avg_cindex:.4f}")
    
    with open(f"{output_dir}/evaluation_results.json", 'w') as f:
        json.dump({
            'avg_pearson': avg_pearson,
            'avg_spearman': avg_spearman,
            'avg_cindex': avg_cindex
        }, f, indent=4)
    
    np.savez(
        f"{output_dir}/predictions.npz", 
        targets=targets, 
        predictions=outputs, 
        biomarkers=biomarkers
    )
    
    return results, targets, outputs

def main():
    data_path = "/playpen/jesse/HE_IF/graph_comparison/cell_feature"
    cell_ids, features, coords, data_splits = load_cell_features(data_path)
    
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
    
    train_indices = data_splits['train']
    val_indices = data_splits['val']
    test_indices = data_splits['test']
    
    biomarker_data = np.log1p(df[biomarker_cols].values)
    
    train_targets = biomarker_data[train_indices]
    val_targets = biomarker_data[val_indices]
    test_targets = biomarker_data[test_indices]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} cells")
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices])
    val_features = scaler.transform(features[val_indices])
    test_features = scaler.transform(features[test_indices])
    
    train_coords = coords[train_indices]
    val_coords = coords[val_indices]
    test_coords = coords[test_indices]

    # Setup cache directory
    cache_dir = "/playpen/jesse/HE_IF/graph_comparison/graph_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use the K-means graph construction with caching
    G_train, train_data = create_kmeans_graph(
        train_features, 
        train_coords, 
        n_clusters=int(np.sqrt(len(train_coords))),
        k_neighbors=15,
        max_distance=50, 
        memory_efficient=True,
        cache_path=cache_dir,
        cache_id="kmeans_train"
    )
    G_val, val_data = create_kmeans_graph(
        val_features, 
        val_coords, 
        n_clusters=int(np.sqrt(len(val_coords))),
        k_neighbors=15,
        max_distance=50, 
        memory_efficient=True,
        cache_path=cache_dir,
        cache_id="kmeans_val"
    )
    G_test, test_data = create_kmeans_graph(
        test_features, 
        test_coords, 
        n_clusters=int(np.sqrt(len(test_coords))),
        k_neighbors=15,
        max_distance=50, 
        memory_efficient=True,
        cache_path=cache_dir,
        cache_id="kmeans_test"
    )

    input_dim = features.shape[1]
    output_dim = len(biomarker_cols)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Use the more complex BiomarkerGNN model
    model = BiomarkerGNN(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=output_dim,
        num_layers=3,
        dropout=0.3,
        residual=True
    )
 
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)
    
    print("Starting training...")

    output_dir = "hybrid_kmeans_gnn_results"
    epochs = 50
    patience = 15
    lr = 0.001
    weight_decay = 1e-5
    batch_size = 256

    model, train_losses, val_losses = train_model(
        model, train_data, val_data,
        train_targets, val_targets,
        num_epochs=epochs,
        patience=patience,
        learning_rate=lr,
        weight_decay=weight_decay,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")

    # Use a larger batch size for evaluation
    eval_batch_size = 128
    results, targets, outputs = evaluate_model(
        model, test_data, biomarker_cols, 
        test_targets,
        output_dir=output_dir,
        batch_size=eval_batch_size
    )
    return model, results

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=7 nohup python k_means_vit.py > ./log/k_means_vit.log 2>&1 &

"""
A488: Pearson=0.0020, Spearman=0.0034, C-index=0.5012
CD3: Pearson=-0.0012, Spearman=0.0065, C-index=0.5022
Ki67: Pearson=-0.0006, Spearman=0.0022, C-index=0.5007
CD4: Pearson=0.0020, Spearman=0.0114, C-index=0.5038
CD20: Pearson=0.0033, Spearman=0.0097, C-index=0.5032
CD163: Pearson=0.0073, Spearman=0.0043, C-index=0.5014
Ecadherin: Pearson=-0.0009, Spearman=0.0058, C-index=0.5019
LaminABC: Pearson=0.0007, Spearman=0.0036, C-index=0.5012
PCNA: Pearson=0.0026, Spearman=0.0066, C-index=0.5022
A555: Pearson=0.0034, Spearman=0.0043, C-index=0.5015
NaKATPase: Pearson=-0.0025, Spearman=0.0018, C-index=0.5006
Keratin: Pearson=-0.0016, Spearman=0.0036, C-index=0.5012
CD45: Pearson=0.0013, Spearman=-0.0041, C-index=0.4986
CD68: Pearson=0.0012, Spearman=0.0059, C-index=0.5020
FOXP3: Pearson=0.0031, Spearman=0.0007, C-index=0.5002
Vimentin: Pearson=0.0029, Spearman=-0.0023, C-index=0.4992
Desmin: Pearson=0.0096, Spearman=0.0054, C-index=0.5018
Ki67_570: Pearson=0.0022, Spearman=0.0058, C-index=0.5020
A647: Pearson=0.0031, Spearman=0.0041, C-index=0.5014
CD45RO: Pearson=-0.0037, Spearman=0.0057, C-index=0.5019
aSMA: Pearson=0.0065, Spearman=0.0046, C-index=0.5015
PD1: Pearson=0.0058, Spearman=0.0107, C-index=0.5036
CD8a: Pearson=0.0041, Spearman=0.0029, C-index=0.5010
PDL1: Pearson=-0.0007, Spearman=0.0017, C-index=0.5006
CDX2: Pearson=0.0033, Spearman=0.0099, C-index=0.5033
CD31: Pearson=0.0111, Spearman=0.0053, C-index=0.5018
Collagen: Pearson=0.0039, Spearman=0.0016, C-index=0.5005

AVERAGE METRICS:
Average Pearson R: 0.0036
Average Spearman R: 0.0048
Average C-index: 0.5015
"""