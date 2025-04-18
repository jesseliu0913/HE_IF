import os
import cv2

os.environ['OPENBLAS_NUM_THREADS'] = '16'  
os.environ['MKL_NUM_THREADS'] = '16' 
os.environ['OMP_NUM_THREADS'] = '16'  
os.environ['NUMEXPR_NUM_THREADS'] = '16'  

import tifffile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.spatial import distance
from sklearn.model_selection import train_test_split


torch.set_num_threads(16)

DATA_PATH = "/playpen/jesse/HIPI/preprocess/data"
he_path = "/playpen/jesse/HIPI/preprocess/data/CRC03-HE.ome.tif"
csv_file = "/playpen/jesse/HIPI/preprocess/data/CRC03_new_coordinates.csv"

def load_single_slice():
    print(f"Loading H&E image: {he_path}")
    he_image = tifffile.imread(he_path)
    
    print(f"Loading cell data: {csv_file}")
    cell_df = pd.read_csv(csv_file)
    
    return he_image, cell_df


def extract_cell_features(he_image, x, y, area, size=224):
    image = np.transpose(he_image, (1, 2, 0))
    H, W = image.shape[:2]

    radius = int(np.sqrt(area / np.pi))
    if radius < 1:
        raise ValueError("`area` is too small - radius < 1 pixel.")

    x_min, x_max = x - radius, x + radius
    y_min, y_max = y - radius, y + radius

    pad_left   = max(0, -x_min)
    pad_right  = max(0,  x_max - W)
    pad_top    = max(0, -y_min)
    pad_bottom = max(0,  y_max - H)

    if pad_left or pad_right or pad_top or pad_bottom:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )
        x_min += pad_left
        x_max += pad_left
        y_min += pad_top
        y_max += pad_top

    patch = image[y_min:y_max, x_min:x_max]
    h, w = patch.shape[:2]
    if h != w:
        diff = abs(h - w)
        if h < w:                 
            top = diff // 2
            bottom = diff - top
            patch = cv2.copyMakeBorder(patch, top, bottom, 0, 0,
                                       cv2.BORDER_CONSTANT, value=0)
        else:                    
            left = diff // 2
            right = diff - left
            patch = cv2.copyMakeBorder(patch, 0, 0, left, right,
                                       cv2.BORDER_CONSTANT, value=0)

    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

    return patch


def construct_graph(cell_features, cell_df, valid_indices, n_clusters=10, k_neighbors=5):
    cell_locations = cell_df.iloc[valid_indices][['X', 'Y']].values
    
    print("Standardizing cell locations...")
    scaler = StandardScaler()
    cell_locations_scaled = scaler.fit_transform(cell_locations)
    
    print(f"Running K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_labels = kmeans.fit_predict(cell_locations_scaled)
    
    print("Creating graph edges...")
    edges = []
    edge_weights = []
    
    for cluster_id in range(n_clusters):
        if cluster_id % 2 == 0:
            print(f"Processing cluster {cluster_id}/{n_clusters}")
            
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) > 1:
            cluster_locations = cell_locations[cluster_indices]
            
            batch_size = 1000
            for i in range(0, len(cluster_indices), batch_size):
                batch_end = min(i + batch_size, len(cluster_indices))
                batch_indices = cluster_indices[i:batch_end]
                batch_locations = cell_locations[batch_indices]
                
                dist_matrix = distance.cdist(batch_locations, cluster_locations, 'euclidean')
                
                for j, idx in enumerate(batch_indices):
                    dist_row = dist_matrix[j]
                    nearest_indices = np.argsort(dist_row)[1:min(k_neighbors+1, len(dist_row))]
                    
                    for k in nearest_indices:
                        if k < len(cluster_indices): 
                            neighbor_idx = cluster_indices[k]
                            edges.append([idx, neighbor_idx])
                            edge_weights.append(1.0 / (dist_row[k] + 1e-6))
    
    if len(edges) == 0:
        print("Warning: No edges created in the graph")
        edges = [[0, 0]]
        edge_weights = [0.0]
    
    print(f"Created {len(edges)} edges. Converting to PyTorch tensors...")
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_weights, dtype=torch.float)
    x = torch.tensor(cell_features, dtype=torch.float)
    y = torch.tensor(cluster_labels, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y), cluster_labels

def create_data_splits(graph_data, cluster_labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    num_nodes = graph_data.x.shape[0]
    print(f"Creating data splits for {num_nodes} nodes...")
    
    train_idx, temp_idx = train_test_split(
        np.arange(num_nodes), 
        train_size=train_ratio,
        stratify=cluster_labels,
        random_state=42
    )
    
    val_size = val_ratio / (test_ratio + val_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=cluster_labels[temp_idx],
        random_state=42
    )
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    graph_data.train_mask = train_mask
    graph_data.val_mask = val_mask
    graph_data.test_mask = test_mask
    
    return graph_data

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)
    
    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        
        x = self.lin(x)
        
        return x

def train_gnn(graph_data, hidden_channels=64, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    graph_data.x = graph_data.x.to(device)
    graph_data.edge_index = graph_data.edge_index.to(device)
    graph_data.edge_attr = graph_data.edge_attr.to(device)
    graph_data.y = graph_data.y.to(device)
    graph_data.train_mask = graph_data.train_mask.to(device)
    graph_data.val_mask = graph_data.val_mask.to(device)
    graph_data.test_mask = graph_data.test_mask.to(device)
    
    num_node_features = graph_data.x.shape[1]
    num_classes = len(torch.unique(graph_data.y))
    print(f"Model input: {num_node_features} features, {num_classes} classes")
    
    model = GNN(num_node_features, hidden_channels, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        loss = criterion(out[graph_data.train_mask], graph_data.y[graph_data.train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_acc = evaluate(model, graph_data, graph_data.train_mask)
            val_acc = evaluate(model, graph_data, graph_data.val_mask)
            test_acc = evaluate(model, graph_data, graph_data.test_mask)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    
    model.load_state_dict(best_model)
    return model

def evaluate(model, graph_data, mask):
    pred = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr).argmax(dim=1)
    correct = (pred[mask] == graph_data.y[mask]).sum().item()
    return correct / mask.sum().item()

def visualize_results(he_image, cell_df, valid_indices, cluster_labels, pred_labels):
    valid_cell_df = cell_df.iloc[valid_indices].copy()
    valid_cell_df['cluster'] = cluster_labels
    valid_cell_df['prediction'] = pred_labels
    
    valid_cell_df.to_csv("CRC03_processed_cells.csv", index=False)
    print("Saved processed cell data to CSV")
    
    print("Creating visualizations...")
    
    # Downsample the H&E image for visualization if it's too large
    max_dim = 2000
    scale_factor = min(max_dim / he_image.shape[1], max_dim / he_image.shape[2])
    if scale_factor < 1:
        from skimage.transform import resize
        viz_image = np.zeros((3, int(he_image.shape[1] * scale_factor), int(he_image.shape[2] * scale_factor)))
        for c in range(3):
            viz_image[c] = resize(he_image[c], viz_image.shape[1:], preserve_range=True)
    else:
        viz_image = he_image
    
    # Visualize cell clusters and predictions on separate plots to reduce memory usage
    fig, ax = plt.subplots(figsize=(12, 10))
    cell_positions = valid_cell_df[['X', 'Y']].values
    scatter = ax.scatter(cell_positions[:, 0], cell_positions[:, 1], 
                         c=valid_cell_df['cluster'], cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    ax.set_title('K-means Clusters')
    plt.tight_layout()
    plt.savefig("CRC03_clusters.png", dpi=150)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(cell_positions[:, 0], cell_positions[:, 1], 
                         c=valid_cell_df['prediction'], cmap='viridis', s=5, alpha=0.7)
    plt.colorbar(scatter, label='Prediction')
    ax.set_title('GNN Predictions')
    plt.tight_layout()
    plt.savefig("CRC03_predictions.png", dpi=150)
    plt.close()
    
    # Save H&E image separately
    rgb_image = np.transpose(viz_image, (1, 2, 0))
    plt.figure(figsize=(12, 10))
    plt.imshow(rgb_image)
    plt.title('H&E Image')
    plt.tight_layout()
    plt.savefig("CRC03_HE_image.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    print("Starting processing...")
    he_image, cell_df = load_single_slice()
    print(f"H&E image shape: {he_image.shape}")
    print(f"Number of cells: {len(cell_df)}")
    
    cell_features, valid_indices = extract_cell_features(he_image, cell_df)
    print(f"Extracted features for {len(cell_features)} cells")
    
    # n_clusters = min(10, len(cell_features) // 100)  # Ensure we don't have too many clusters
    # graph_data, cluster_labels = construct_graph(cell_features, cell_df, valid_indices, n_clusters=n_clusters)
    # print(f"Constructed graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges")
    
    # graph_data = create_data_splits(graph_data, cluster_labels)
    # print(f"Train: {graph_data.train_mask.sum().item()}, "
    #       f"Val: {graph_data.val_mask.sum().item()}, "
    #       f"Test: {graph_data.test_mask.sum().item()}")
    
    # model = train_gnn(graph_data, hidden_channels=32, num_epochs=50)  # Reduced hidden channels and epochs
    # print("Trained GNN model")
    
    # # Clear CUDA cache if using GPU
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    
    # # Get predictions
    # device = next(model.parameters()).device
    # model.eval()
    # with torch.no_grad():
    #     pred = model(graph_data.x.to(device), 
    #                  graph_data.edge_index.to(device), 
    #                  graph_data.edge_attr.to(device)).argmax(dim=1).cpu().numpy()
    
    # visualize_results(he_image, cell_df, valid_indices, cluster_labels, pred)
    # print("Saved visualization results")