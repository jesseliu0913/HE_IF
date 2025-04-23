import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

class VitPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(VitPredictionModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = F.relu(self.bn_input(x))
        
        orig_x = x
        
        for i, layer in enumerate(self.layers):
            x_new = layer(x)
            x_new = F.relu(self.bns[i](x_new))
            
            if i % 2 == 1:
                x = x + x_new  
            else:
                x = x_new
        
        x = x + orig_x  
        x = self.output(x)
        
        return x

def load_cell_features(features_path, data_splits_path):
    feature_data = np.load(features_path)
    cell_ids = feature_data['cell_ids']
    features = feature_data['features']
    coords = feature_data['coords']
    
    with open(data_splits_path, 'r') as f:
        data_splits = json.load(f)
    
    return cell_ids, features, coords, data_splits

def train_model(model, train_data, train_targets, val_data, val_targets, device, num_epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.HuberLoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(train_data)
        loss = criterion(outputs, train_targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_data)
            val_loss = criterion(val_outputs, val_targets).item()
            val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_model)
    return model, train_losses, val_losses

def evaluate_model(model, test_data, test_targets, biomarkers):
    model.eval()
    
    with torch.no_grad():
        outputs = model(test_data)
    
    targets = test_targets.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    results = {}
    
    for i, biomarker in enumerate(biomarkers):
        pearson_r, _ = pearsonr(targets[:, i], outputs[:, i])
        spearman_r, _ = spearmanr(targets[:, i], outputs[:, i])
        
        actual_orig = np.expm1(targets[:, i])
        pred_orig = np.expm1(np.clip(outputs[:, i], -10, 10))
        c_index = concordance_index(actual_orig, pred_orig)
        
        results[biomarker] = {
            'PearsonR': pearson_r,
            'SpearmanR': spearman_r,
            'C-index': c_index
        }
        
        print(f"{biomarker}: Pearson={pearson_r:.4f}, Spearman={spearman_r:.4f}, C-index={c_index:.4f}")
    
    avg_pearson = np.mean([metrics['PearsonR'] for metrics in results.values()])
    avg_spearman = np.mean([metrics['SpearmanR'] for metrics in results.values()])
    avg_cindex = np.mean([metrics['C-index'] for metrics in results.values()])
    
    print(f"Average Pearson R: {avg_pearson:.4f}")
    print(f"Average Spearman R: {avg_spearman:.4f}")
    print(f"Average C-index: {avg_cindex:.4f}")
    
    cell_results = []
    for i in range(targets.shape[0]):
        row = {}
        row['CellIndex'] = i
            
        for j, biomarker in enumerate(biomarkers):
            row[f"{biomarker}_Target"] = np.expm1(targets[i, j])
            row[f"{biomarker}_Prediction"] = np.expm1(np.clip(outputs[i, j], -10, 10))
            
        cell_results.append(row)
    
    cell_results_df = pd.DataFrame(cell_results)
    
    return results, targets, outputs, cell_results_df

def main():
    features_path = "./cell_feature/cell_features.npz"
    data_splits_path = "./cell_feature/data_splits.json"
    csv_file = "/playpen/jesse/HIPI/preprocess/data/CRC03_new_coordinates.csv"
    output_path = "./vit_prediction"
    
    os.makedirs(output_path, exist_ok=True)
    
    cell_ids, features, coords, data_splits = load_cell_features(features_path, data_splits_path)
    
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
    
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)} cells")
    
    scaler = StandardScaler()
    train_features = scaler.fit_transform(features[train_indices])
    val_features = scaler.transform(features[val_indices])
    test_features = scaler.transform(features[test_indices])
    
    train_targets = biomarker_data[train_indices]
    val_targets = biomarker_data[val_indices]
    test_targets = biomarker_data[test_indices]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data = torch.FloatTensor(train_features).to(device)
    train_targets = torch.FloatTensor(train_targets).to(device)
    
    val_data = torch.FloatTensor(val_features).to(device)
    val_targets = torch.FloatTensor(val_targets).to(device)
    
    test_data = torch.FloatTensor(test_features).to(device)
    test_targets = torch.FloatTensor(test_targets).to(device)
    
    input_dim = train_features.shape[1]
    hidden_dim = 128
    output_dim = len(biomarker_cols)
    
    model = VitPredictionModel(input_dim, hidden_dim, output_dim).to(device)
    
    print("Starting training...")
    model, train_losses, val_losses = train_model(model, train_data, train_targets, val_data, val_targets, device, num_epochs=50)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'vit_loss_curves.png'))
    
    print("Evaluating model...")
    results, targets, outputs, cell_results_df = evaluate_model(model, test_data, test_targets, biomarker_cols)
    cell_results_df.to_csv(os.path.join(output_path, "raw_vit_results.csv"), index=False)
    
    with open(os.path.join(output_path, 'vit_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    torch.save(model.state_dict(), os.path.join(output_path, "vit_model.pt"))
    np.savez(os.path.join(output_path, "vit_predictions.npz"), targets=targets, predictions=outputs, biomarkers=biomarker_cols)
    
    print(f"Model saved to {os.path.join(output_path, 'vit_model.pt')}")
    print(f"Results saved to {os.path.join(output_path, 'raw_vit_results.csv')}")
    print(f"Metrics saved to {os.path.join(output_path, 'vit_metrics.json')}")

if __name__ == "__main__":
    main()


"""
Average Pearson R: -0.0042
Average Spearman R: -0.0036
Average C-index: 0.4990
"""