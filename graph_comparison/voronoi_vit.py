import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import Data
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from scipy.spatial import Voronoi, voronoi_plot_2d
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

def create_voronoi_graph(cell_features, cell_coords, max_distance=50, memory_efficient=True, 
                         cache_path=None, cache_id=None):
    """
    Create graph based on Voronoi diagram with caching capability
    """
    # If cache path and ID are provided, try to load from cache
    if cache_path and cache_id:
        cache_file = f"{cache_path}/voronoi_graph_cache_{cache_id}.pkl"
        if os.path.exists(cache_file):
            print(f"Loading cached graph from {cache_file}...")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                G = cached_data["G"]
                data = cached_data["data"]
                return G, data
    
    print(f"Creating Voronoi graph for {len(cell_coords)} cells...")
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in tqdm(range(len(cell_coords)), desc="Adding nodes"):
        G.add_node(i, pos=cell_coords[i])
    
    # Compute Voronoi diagram
    # Add a small amount of jitter to coordinates to avoid issues with duplicate points
    jittered_coords = cell_coords + np.random.normal(0, 1e-10, cell_coords.shape)
    vor = Voronoi(jittered_coords)
    
    # Create edges between adjacent Voronoi cells
    print("Creating edges between adjacent Voronoi cells...")
    ridge_points = vor.ridge_points
    for i in tqdm(range(len(ridge_points)), desc="Adding edges"):
        p1, p2 = ridge_points[i]
        # Only add edge if the distance is below max_distance
        if max_distance is None or np.linalg.norm(cell_coords[p1] - cell_coords[p2]) <= max_distance:
            G.add_edge(p1, p2)
    
    # For points at the boundary of the Voronoi diagram (with no neighbors or very few),
    # we need to ensure they have connections. Use KNN for these points.
    degree_dict = dict(G.degree())
    isolated_nodes = [node for node, degree in degree_dict.items() if degree < 2]
    
    if isolated_nodes:
        print(f"Adding connections for {len(isolated_nodes)} isolated or boundary nodes...")
        isolated_coords = cell_coords[isolated_nodes]
        
        # Use KNN to find nearest neighbors for isolated points
        k = min(5, len(cell_coords) - 1)  # Find at least 5 neighbors if possible
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(cell_coords)  # +1 because the point itself is included
        distances, indices = nbrs.kneighbors(isolated_coords)
        
        for i, node_idx in enumerate(isolated_nodes):
            for j in range(1, k+1):  # Skip the first neighbor (the point itself)
                neighbor_idx = indices[i, j]
                if max_distance is None or distances[i, j] <= max_distance:
                    G.add_edge(node_idx, neighbor_idx)
    
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
        cache_file = f"{cache_path}/voronoi_graph_cache_{cache_id}.pkl"
        print(f"Saving graph to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump({"G": G, "data": data}, f)
    
    return G, data

# The rest of your code remains unchanged
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
    
    # Use the Voronoi graph construction with caching
    G_train, train_data = create_voronoi_graph(
        train_features, 
        train_coords, 
        max_distance=50, 
        memory_efficient=True,
        cache_path=cache_dir,
        cache_id="voronoi_train"
    )
    G_val, val_data = create_voronoi_graph(
        val_features, 
        val_coords, 
        max_distance=50, 
        memory_efficient=True,
        cache_path=cache_dir,
        cache_id="voronoi_val"
    )
    G_test, test_data = create_voronoi_graph(
        test_features, 
        test_coords, 
        max_distance=50, 
        memory_efficient=True,
        cache_path=cache_dir,
        cache_id="voronoi_test"
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

    output_dir = "voronoi_gnn_results"
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

# CUDA_VISIBLE_DEVICES=5 nohup python voronoi_vit.py > voronoi_vit.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python voronoi_vit.py


"""
Hoechst1: Pearson=0.0006, Spearman=-0.0035, C-index=0.4988
Hoechst2: Pearson=0.0004, Spearman=-0.0013, C-index=0.4997
Hoechst3: Pearson=-0.0013, Spearman=-0.0037, C-index=0.4988
Hoechst4: Pearson=-0.0021, Spearman=-0.0019, C-index=0.4994
Hoechst5: Pearson=-0.0036, Spearman=-0.0026, C-index=0.4998
Hoechst6: Pearson=-0.0029, Spearman=-0.0017, C-index=0.4994
Hoechst7: Pearson=-0.0026, Spearman=-0.0002, C-index=0.4999
Hoechst8: Pearson=-0.0009, Spearman=-0.0010, C-index=0.4997
Hoechst9: Pearson=0.0010, Spearman=-0.0016, C-index=0.4995
A488: Pearson=-0.0029, Spearman=-0.0022, C-index=0.4993
CD3: Pearson=0.0004, Spearman=0.0016, C-index=0.5005
Ki67: Pearson=-0.0008, Spearman=-0.0027, C-index=0.4991
CD4: Pearson=0.0017, Spearman=0.0032, C-index=0.5011
CD20: Pearson=-0.0010, Spearman=0.0022, C-index=0.5007
CD163: Pearson=-0.0015, Spearman=0.0013, C-index=0.5004
Ecadherin: Pearson=-0.0038, Spearman=0.0033, C-index=0.5011
LaminABC: Pearson=0.0007, Spearman=0.0065, C-index=0.5022
PCNA: Pearson=0.0022, Spearman=0.0008, C-index=0.5003
A555: Pearson=-0.0013, Spearman=-0.0041, C-index=0.4986
NaKATPase: Pearson=-0.0032, Spearman=0.0029, C-index=0.5010
Keratin: Pearson=-0.0013, Spearman=0.0025, C-index=0.5008
CD45: Pearson=0.0034, Spearman=0.0011, C-index=0.5004
CD68: Pearson=-0.0057, Spearman=0.0017, C-index=0.5005
FOXP3: Pearson=-0.0006, Spearman=-0.0011, C-index=0.4997
Vimentin: Pearson=0.0010, Spearman=-0.0004, C-index=0.4999
Desmin: Pearson=0.0014, Spearman=0.0100, C-index=0.5033
Ki67_570: Pearson=0.0045, Spearman=-0.0029, C-index=0.4990
A647: Pearson=-0.0039, Spearman=0.0036, C-index=0.5012
CD45RO: Pearson=0.0012, Spearman=0.0045, C-index=0.5015
aSMA: Pearson=0.0020, Spearman=0.0057, C-index=0.5019
PD1: Pearson=-0.0053, Spearman=0.0134, C-index=0.5045
CD8a: Pearson=-0.0041, Spearman=0.0019, C-index=0.5006
PDL1: Pearson=-0.0078, Spearman=0.0059, C-index=0.5020
CDX2: Pearson=-0.0053, Spearman=0.0031, C-index=0.5010
CD31: Pearson=-0.0006, Spearman=0.0060, C-index=0.5020
Collagen: Pearson=0.0012, Spearman=0.0054, C-index=0.5018

AVERAGE METRICS:
Average Pearson R: -0.0011
Average Spearman R: 0.0015
Average C-index: 0.5005
"""