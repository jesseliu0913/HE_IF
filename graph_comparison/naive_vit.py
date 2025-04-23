import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
import pickle
warnings.filterwarnings('ignore')

# Set environment variables to limit thread usage
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

def load_cell_features(data_path="/playpen/jesse/HE_IF/graph_comparison/cell_feature"):
    """Load cell features, coordinates, and data splits."""
    feature_data = np.load(f"{data_path}/cell_features.npz")
    cell_ids = feature_data['cell_ids']
    features = feature_data['features']
    coords = feature_data['coords']
    
    with open(f"{data_path}/data_splits.json", 'r') as f:
        data_splits = json.load(f)
    
    return cell_ids, features, coords, data_splits

class BiomarkerMLP(nn.Module):
    """
    Multi-layer perceptron for biomarker prediction from cell features.
    This replaces the GNN model with a direct MLP approach.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, num_layers=3, dropout=0.3):
        super(BiomarkerMLP, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.BatchNorm1d(hidden_dim)
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim * 2))
                self.norms.append(nn.BatchNorm1d(hidden_dim * 2))
            elif i == num_layers - 1:
                self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim * 2, hidden_dim * 2))
                self.norms.append(nn.BatchNorm1d(hidden_dim * 2))
        
        self.dropout = nn.Dropout(dropout)
        
        self.pre_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        
        self.predictor = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Process through hidden layers
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
        
        # Final layers
        x = self.pre_output(x)
        x = self.predictor(x)
        
        return x

def train_model(model, train_features, val_features, train_targets, val_targets, num_epochs=100, 
                patience=15, learning_rate=1e-3, weight_decay=1e-5, output_dir="results_naive", batch_size=256):
    """Train the MLP model using features directly."""
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert features and targets to PyTorch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    val_features = torch.tensor(val_features, dtype=torch.float32)
    
    # Move validation data to device
    val_features = val_features.to(device)
    val_targets = val_targets.to(device)
    model = model.to(device)
    
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
    
    num_samples = train_features.size(0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Shuffle indices for this epoch
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:min(i+batch_size, num_samples)]
            
            # Get batch data
            x_batch = train_features[batch_indices].to(device)
            y_batch = train_targets[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            try:
                out = model(x_batch)
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
            train_loss = total_loss / num_samples
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                out = model(val_features)
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

def evaluate_model(model, test_features, biomarkers, test_targets, output_dir="results_naive", batch_size=512):
    """Evaluate the model on test data."""
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert test features to tensor
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_features = test_features.to(device)
    
    model.eval()
    
    # Process the test data in batches
    num_samples = test_features.size(0)
    all_outputs = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_features = test_features[i:end_idx]
            
            batch_out = model(batch_features)
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
    """Main function to run the biomarker prediction pipeline."""
    data_path = "/playpen/jesse/HE_IF/graph_comparison/cell_feature"
    cell_ids, features, coords, data_splits = load_cell_features(data_path)
    
    csv_file = "/playpen/jesse/HE_IF/graph_comparison/cell_feature/CRC03_new_coordinates.csv"
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
    
    # Standardize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices])
    val_features = scaler.transform(features[val_indices])
    test_features = scaler.transform(features[test_indices])
    
    input_dim = features.shape[1]
    output_dim = len(biomarker_cols)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create MLP model instead of GNN
    model = BiomarkerMLP(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        num_layers=3,
        dropout=0.3
    )
 
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    val_targets = torch.tensor(val_targets, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)
    
    print("Starting training...")

    output_dir = "mlp_biomarker_results"
    epochs = 50
    patience = 15
    lr = 0.001
    weight_decay = 1e-5
    batch_size = 1024

    model, train_losses, val_losses = train_model(
        model, train_features, val_features,
        train_targets, val_targets,
        num_epochs=epochs,
        patience=patience,
        learning_rate=lr,
        weight_decay=weight_decay,
        output_dir=output_dir,
        batch_size=batch_size
    )
    
    torch.save(model.state_dict(), f"{output_dir}/final_model.pt")

    eval_batch_size = 512
    results, targets, outputs = evaluate_model(
        model, test_features, biomarker_cols, 
        test_targets,
        output_dir=output_dir,
        batch_size=eval_batch_size
    )
    return model, results

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=5 nohup python naive_vit.py > ./log/naive_vit.log 2>&1 &