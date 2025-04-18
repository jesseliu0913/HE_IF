# import os
# import json
# import tifffile
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


# DATA_PATH = "/playpen/jesse/HIPI/preprocess/data"
# he_files = []
# for root, dirs, files in os.walk(DATA_PATH):
#     for file in files:
#         if file.endswith('.ome.tif') and 'HE' in file:
#             full_path = os.path.join(root, file)
#             he_files.append(full_path)

# # shape of HE image (3, 41794, 97089)
# """
# all info in csv file:
# ['Hoechst1', 'Hoechst2', 'Hoechst3', 'Hoechst4', 'Hoechst5', 'Hoechst6',
#        'Hoechst7', 'Hoechst8', 'Hoechst9', 'A488', 'CD3', 'Ki67', 'CD4',
#        'CD20', 'CD163', 'Ecadherin', 'LaminABC', 'PCNA', 'A555', 'NaKATPase',
#        'Keratin', 'CD45', 'CD68', 'FOXP3', 'Vimentin', 'Desmin', 'Ki67_570',
#        'A647', 'CD45RO', 'aSMA', 'PD1', 'CD8a', 'PDL1', 'CDX2', 'CD31',
#        'Collagen', 'AREA', 'CIRC', 'X', 'Y', 'frame', 'COL', 'ROW', 'Xt',
#        'Yt']
# """
# for he_path in he_files:
#     file_name = he_path.replace(DATA_PATH, '').split("-")[0].replace("/", "")
#     # he_image = tifffile.imread(he_path)
#     csv_file = os.path.join(DATA_PATH, file_name + ".csv")
#     df = pd.read_csv(csv_file)
#     # print(he_image.shape)
#     print(df.columns)


import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(GNNModel, self).__init__()
        self.num_layers = num_layers
        
        # Initial convolutional layer
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        
        # Additional convolutional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Apply GNN layers
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final prediction
        x = self.out(x)
        
        return x




def main():
    # Load cell features
    data_path = "/playpen/jesse/HE_IF/graph_comparison/cell_feature"
    cell_ids, features, coords, data_splits = load_cell_features(data_path)
    
    print(f"Loaded {len(cell_ids)} cells with {features.shape[1]} features each")
    
    # Construct KNN graph
    edges, edge_weights, adj_matrix = construct_knn_graph(features, coords, k=20, use_coords=False)
    
    # Also create spatial graph
    spatial_edges, spatial_weights, spatial_adj = construct_knn_graph(features, coords, k=10, use_coords=True)
    
    # Load biomarker data
    csv_file = "/playpen/jesse/HIPI/preprocess/data/CRC03_new_coordinates.csv"
    aligned_features, aligned_coords, aligned_targets, biomarker_cols, aligned_indices = load_and_preprocess_data(
        csv_file, cell_ids, features, coords)
    
    print(f"Aligned data: {len(aligned_indices)} cells with {len(biomarker_cols)} biomarkers")
    
    # Split data for training
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        aligned_features, aligned_targets, aligned_indices, test_size=0.2, random_state=42)
    
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_train, y_train, train_idx, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Create PyTorch Geometric Data objects
    train_data = prepare_pytorch_geometric_data(X_train, edges, edge_weights, y_train, train_idx)
    val_data = prepare_pytorch_geometric_data(X_val, edges, edge_weights, y_val, val_idx)
    test_data = prepare_pytorch_geometric_data(X_test, edges, edge_weights, y_test, test_idx)
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=1)
    val_loader = DataLoader([val_data], batch_size=1)
    test_loader = DataLoader([test_data], batch_size=1)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_train.shape[1]
    hidden_dim = 128
    output_dim = len(biomarker_cols)
    
    model = GNNModel(input_dim, hidden_dim, output_dim).to(device)
    
    # Train model
    model = train_model(model, train_loader, val_loader, device, num_epochs=100)
    
    # Evaluate model
    results, all_targets, all_outputs = evaluate_model(model, test_loader, device, biomarker_cols)
    
    # Print results
    print("\nBiomarker Prediction Results:")
    for biomarker, metrics in results.items():
        print(f"{biomarker}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Calculate overall metrics
    avg_pearson = np.mean([metrics['PearsonR'] for metrics in results.values()])
    avg_spearman = np.mean([metrics['SpearmanR'] for metrics in results.values()])
    avg_cindex = np.mean([metrics['C-index'] for metrics in results.values()])
    
    print("\nOverall Performance:")
    print(f"Average Pearson R: {avg_pearson:.4f}")
    print(f"Average Spearman R: {avg_spearman:.4f}")
    print(f"Average C-index: {avg_cindex:.4f}")
    
    # Save model and results
    torch.save(model.state_dict(), "biomarker_gnn_model.pt")
    np.savez("prediction_results.npz", 
             targets=all_targets, 
             predictions=all_outputs, 
             biomarkers=biomarker_cols)
    
    return model, results

if __name__ == "__main__":
    main()