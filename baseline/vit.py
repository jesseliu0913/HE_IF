import os
import time
from datetime import timedelta
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import tifffile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import cv2


class HEBiomarkerDataset(Dataset):
    def __init__(self, df, he_image, biomarkers, morphology_features):
        t_start = time.time()
        
        self.df = df
        self.he_image = he_image
        self.biomarkers = biomarkers
        self.patch_size = 224
        
        t_feature_check_start = time.time()
        available_features = []
        for feature in morphology_features:
            if feature in df.columns:
                available_features.append(feature)
        
        if not available_features:
            print("Warning: No morphology features found in dataset! Using only Area and centroid coordinates.")
            self.morphology_features = ["Area", "X_centroid", "Y_centroid"]
        else:
            print(f"Using {len(available_features)} morphology features: {available_features}")
            self.morphology_features = available_features
        t_feature_check_end = time.time()
        print(f"[TIME] Feature checking: {t_feature_check_end - t_feature_check_start:.4f}s")
            
        t_scaler_start = time.time()
        self.scaler = StandardScaler()
        morphology_data = df[self.morphology_features].values
        self.scaler.fit(morphology_data)
        t_scaler_end = time.time()
        print(f"[TIME] StandardScaler fitting: {t_scaler_end - t_scaler_start:.4f}s")
        
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        t_end = time.time()
        print(f"[TIME] Total dataset initialization: {t_end - t_start:.4f}s")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        t_getitem_start = time.time()
        
        row = self.df.iloc[idx]
        x, y, area = int(row["X_centroid"]), int(row["Y_centroid"]), int(row["Area"])
        radius = int(np.sqrt(area / np.pi))
        radius = max(radius, self.patch_size // 4)
        
        patch = self._extract_patch(x, y, radius)
        patch_tensor = self._transform_patch(patch)
        morphology_tensor = self._get_morphology_features(row)
        target_tensor = self._get_biomarker_targets(row)
        
        return patch_tensor, morphology_tensor, target_tensor
    
    def _extract_patch(self, x, y, radius):
        x_min, x_max = max(0, x - radius), min(self.he_image.shape[1], x + radius)
        y_min, y_max = max(0, y - radius), min(self.he_image.shape[0], y + radius)
        
        if x_max <= x_min or y_max <= y_min or x_min >= self.he_image.shape[1] or y_min >= self.he_image.shape[0]:
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
        
        patch = self.he_image[y_min:y_max, x_min:x_max]
        
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            return np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
        
        resized_patch = cv2.resize(patch, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        return resized_patch
    
    def _transform_patch(self, patch):
        patch = patch.astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        patch_tensor = torch.tensor(patch, dtype=torch.float32)
        patch_tensor = self.transform(patch_tensor)
        return patch_tensor
    
    def _get_morphology_features(self, row):
        morphology_values = [row[feature] for feature in self.morphology_features]
        morphology_values = np.array(morphology_values, dtype=np.float32).reshape(1, -1)
        morphology_values = self.scaler.transform(morphology_values).flatten()
        return torch.tensor(morphology_values, dtype=torch.float32)
    
    def _get_biomarker_targets(self, row):
        targets = np.zeros(len(self.biomarkers), dtype=np.float32)
        for i, biomarker in enumerate(self.biomarkers):
            targets[i] = np.log1p(row[biomarker])
        return torch.tensor(targets, dtype=torch.float32)


class HEBiomarkerModel(nn.Module):
    def __init__(self, morphology_dim, hidden_dim, output_dim):
        super(HEBiomarkerModel, self).__init__()
        
        t_init_start = time.time()
        
        t_backbone_start = time.time()
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()
        t_backbone_end = time.time()
        print(f"[TIME] ViT backbone initialization: {t_backbone_end - t_backbone_start:.4f}s")
        
        t_morph_start = time.time()
        self.morph_encoder = nn.Sequential(
            nn.Linear(morphology_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        t_morph_end = time.time()
        print(f"[TIME] Morphology encoder initialization: {t_morph_end - t_morph_start:.4f}s")
        
        t_fusion_start = time.time()

        self.fusion = nn.Sequential(
            nn.Linear(768 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        t_fusion_end = time.time()
        print(f"[TIME] Fusion layer initialization: {t_fusion_end - t_fusion_start:.4f}s")
        
        t_output_start = time.time()
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        t_output_end = time.time()
        print(f"[TIME] Output layer initialization: {t_output_end - t_output_start:.4f}s")
        
        t_init_end = time.time()
        print(f"[TIME] Total model initialization: {t_init_end - t_init_start:.4f}s")
    
    def forward(self, images, morphology):
        t_forward_start = time.time()
        
        t_backbone_forward_start = time.time()
        img_features = self.backbone(images)
        t_backbone_forward_end = time.time()
        
        t_morph_forward_start = time.time()
        morph_features = self.morph_encoder(morphology)
        t_morph_forward_end = time.time()
        
        t_fusion_forward_start = time.time()
        combined_features = torch.cat([img_features, morph_features], dim=1)
        features = self.fusion(combined_features)
        output = self.output_layer(features)
        t_fusion_forward_end = time.time()
        
        t_forward_end = time.time()
        
        if not self.training and random.random() < 0.01:
            print(f"[TIME] Forward pass breakdown: "
                  f"Total: {t_forward_end - t_forward_start:.4f}s, "
                  f"Backbone: {t_backbone_forward_end - t_backbone_forward_start:.4f}s, "
                  f"Morph encoder: {t_morph_forward_end - t_morph_forward_start:.4f}s, "
                  f"Fusion+Output: {t_fusion_forward_end - t_fusion_forward_start:.4f}s")
        
        return output


def find_data_files(data_directory):
    t_start = time.time()
    
    folders = [os.path.join(data_directory, d) for d in os.listdir(data_directory) 
               if os.path.isdir(os.path.join(data_directory, d))]

    if not folders:
        raise ValueError("No valid folders found in the data directory")
    
    print(f"Found {len(folders)} folders")
    
    all_data_files = []
    for folder in folders:
        file_index = os.path.basename(folder)
        csv_file = None
        he_path = None
        
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if file.endswith('.csv') and file_index in file:
                csv_file = file_path
            elif 'registered.ome.tif' in file:
                he_path = file_path
        
        if csv_file is not None and he_path is not None:
            all_data_files.append((csv_file, he_path, file_index))
    
    if not all_data_files:
        raise ValueError(f"No valid data files found in any folder")
    
    print(f"Found {len(all_data_files)} valid data file pairs across all folders")
    
    t_end = time.time()
    print(f"[TIME] File search: {t_end - t_start:.4f}s")
    
    return all_data_files


def load_data(csv_file, he_path):
    t_csv_start = time.time()
    df = pd.read_csv(csv_file)
    t_csv_end = time.time()
    print(f"[TIME] CSV loading: {t_csv_end - t_csv_start:.4f}s")
    
    t_image_start = time.time()
    he_image = tifffile.imread(he_path)
    t_image_end = time.time()
    print(f"[TIME] TIFF image loading: {t_image_end - t_image_start:.4f}s")
    print(f"Image shape: {he_image.shape}, dtype: {he_image.dtype}, size: {he_image.nbytes / (1024**2):.2f} MB")
    
    return df, he_image


def create_data_splits(all_data_files, test_size=0.1, val_size=0.1, random_state=42):
    t_start = time.time()
    
    n_files = len(all_data_files)
    indices = np.arange(n_files)
    
    train_val_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    val_size_adjusted = val_size / (1 - test_size)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size_adjusted, random_state=random_state)
    
    train_files = [all_data_files[i] for i in train_indices]
    val_files = [all_data_files[i] for i in val_indices]
    test_files = [all_data_files[i] for i in test_indices]
    
    print(f"Train set: {len(train_files)} slices")
    print(f"Validation set: {len(val_files)} slices")
    print(f"Test set: {len(test_files)} slices")
    
    t_end = time.time()
    print(f"[TIME] Data splitting: {t_end - t_start:.4f}s")
    
    return train_files, val_files, test_files


def create_datasets_and_loaders(data_files, biomarkers, morphology_features, batch_size=3):
    t_dataset_start = time.time()
    
    datasets = []
    for csv_file, he_path, file_index in data_files:
        df, he_image = load_data(csv_file, he_path)
        dataset = HEBiomarkerDataset(df, he_image, biomarkers, morphology_features)
        datasets.append(dataset)
    
    if not datasets:
        raise ValueError("No datasets created. Check data files.")
    
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    t_dataset_end = time.time()
    print(f"[TIME] Dataset creation: {t_dataset_end - t_dataset_start:.4f}s")
    
    t_dataloader_start = time.time()
    data_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=8, 
                             pin_memory=True, persistent_workers=True, drop_last=True)
    t_dataloader_end = time.time()
    print(f"[TIME] DataLoader creation: {t_dataloader_end - t_dataloader_start:.4f}s")
    
    return combined_dataset, data_loader


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch, num_epochs, batch_checkpoints_dir):
    t_start = time.time()
    
    model.train()
    train_loss = 0.0
    all_losses = []
    
    batch_times = []
    data_loading_times = []
    to_device_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for i, (images, morphology, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
        t_batch_start = time.time()
        t_data_load_end = time.time()
        data_loading_times.append(t_data_load_end - t_batch_start)
        
        t_to_device_start = time.time()
        images = images.to(device, non_blocking=True)
        morphology = morphology.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        t_to_device_end = time.time()
        to_device_times.append(t_to_device_end - t_to_device_start)
        
        # Skip this batch if it's a singleton (should not happen with drop_last=True)
        if images.size(0) < 2:
            print(f"WARNING: Skipping batch {i} with only {images.size(0)} samples")
            continue
        
        optimizer.zero_grad(set_to_none=True)
        
        t_forward_start = time.time()
        with torch.amp.autocast('cuda'):
            outputs = model(images, morphology)
            loss = criterion(outputs, targets)
        t_forward_end = time.time()
        forward_times.append(t_forward_end - t_forward_start)
        
        t_backward_start = time.time()
        scaler.scale(loss).backward()
        t_backward_end = time.time()
        backward_times.append(t_backward_end - t_backward_start)
        
        t_optimizer_start = time.time()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        t_optimizer_end = time.time()
        optimizer_times.append(t_optimizer_end - t_optimizer_start)
        
        batch_loss = loss.item()
        train_loss += batch_loss * images.size(0)
        all_losses.append(batch_loss)
        
        t_batch_end = time.time()
        batch_times.append(t_batch_end - t_batch_start)
        
        wandb.log({
            "batch": i + epoch * len(train_loader),
            "batch_loss": batch_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "batch_time": batch_times[-1],
            "data_loading_time": data_loading_times[-1],
            "forward_time": forward_times[-1],
            "backward_time": backward_times[-1],
            "optimizer_time": optimizer_times[-1]
        })
        
        # checkpoint_path = os.path.join(batch_checkpoints_dir, f"epoch{epoch+1}_batch{i+1}.pt")
        # model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        # torch.save({
        #     'epoch': epoch,
        #     'batch': i,
        #     'model_state_dict': model_state_dict,
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': batch_loss,
        #     'batch_losses': all_losses
        # }, checkpoint_path)
        
        if i == 0 or (i + 1) % 10 == 0:
            print(f"[TIME] Batch {i+1} breakdown - "
                  f"Total: {batch_times[-1]:.4f}s, "
                  f"Data loading: {data_loading_times[-1]:.4f}s, "
                  f"To device: {to_device_times[-1]:.4f}s, "
                  f"Forward: {forward_times[-1]:.4f}s, "
                  f"Backward: {backward_times[-1]:.4f}s, "
                  f"Optimizer: {optimizer_times[-1]:.4f}s, "
                  f"Loss: {batch_loss:.6f}")
    
    if len(train_loader.dataset) > 0 and len(all_losses) > 0:
        train_loss /= len(train_loader.dataset)
        t_end = time.time()
        
        plt.figure(figsize=(10, 6))
        plt.plot(all_losses)
        plt.title(f'Epoch {epoch+1} Batch-wise Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(f'epoch_{epoch+1}_loss.png')
        plt.close()
        
        wandb.log({
            "epoch": epoch,
            "epoch_train_loss": train_loss,
            "epoch_loss_plot": wandb.Image(f'epoch_{epoch+1}_loss.png')
        })
        
        print(f"[TIME] Training stats - "
              f"Avg batch time: {np.mean(batch_times):.4f}s, "
              f"Avg data loading time: {np.mean(data_loading_times):.4f}s, "
              f"Avg to device time: {np.mean(to_device_times):.4f}s, "
              f"Avg forward time: {np.mean(forward_times):.4f}s, "
              f"Avg backward time: {np.mean(backward_times):.4f}s, "
              f"Avg optimizer time: {np.mean(optimizer_times):.4f}s")
    else:
        train_loss = float('inf')
        t_end = time.time()
        all_losses = []
        print("WARNING: No batches were processed in this epoch!")
    
    return train_loss, all_losses, t_end - t_start


def validate(model, val_loader, criterion, device, epoch, num_epochs):
    t_start = time.time()
    
    model.eval()
    val_loss = 0.0
    all_val_losses = []
    
    val_batch_times = []
    val_data_loading_times = []
    val_to_device_times = []
    val_forward_times = []
    
    with torch.no_grad():
        for i, (images, morphology, targets) in enumerate(tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{num_epochs}")):
            t_val_batch_start = time.time()
            t_val_data_load_end = time.time()
            val_data_loading_times.append(t_val_data_load_end - t_val_batch_start)
            
            t_val_to_device_start = time.time()
            images = images.to(device, non_blocking=True)
            morphology = morphology.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            t_val_to_device_end = time.time()
            val_to_device_times.append(t_val_to_device_end - t_val_to_device_start)
            
            # Skip singleton batches during validation too
            if images.size(0) < 2:
                print(f"WARNING: Skipping validation batch {i} with only {images.size(0)} samples")
                continue
            
            t_val_forward_start = time.time()
            with torch.amp.autocast('cuda'):
                outputs = model(images, morphology)
                loss = criterion(outputs, targets)
            t_val_forward_end = time.time()
            val_forward_times.append(t_val_forward_end - t_val_forward_start)
            
            batch_loss = loss.item()
            val_loss += batch_loss * images.size(0)
            all_val_losses.append(batch_loss)
            
            wandb.log({
                "val_batch": i + epoch * len(val_loader),
                "val_batch_loss": batch_loss,
            })
            
            t_val_batch_end = time.time()
            val_batch_times.append(t_val_batch_end - t_val_batch_start)
            
            if i == 0 or (i + 1) % 10 == 0:
                print(f"[TIME] Validation batch {i+1} breakdown - "
                      f"Total: {val_batch_times[-1]:.4f}s, "
                      f"Data loading: {val_data_loading_times[-1]:.4f}s, "
                      f"To device: {val_to_device_times[-1]:.4f}s, "
                      f"Forward: {val_forward_times[-1]:.4f}s, "
                      f"Loss: {batch_loss:.6f}")
    
    # Check if we processed any batches
    if len(val_loader.dataset) > 0 and len(all_val_losses) > 0:
        val_loss /= len(val_loader.dataset)
        t_end = time.time()
        
        plt.figure(figsize=(10, 6))
        plt.plot(all_val_losses)
        plt.title(f'Epoch {epoch+1} Validation Batch-wise Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.savefig(f'epoch_{epoch+1}_val_loss.png')
        plt.close()
        
        wandb.log({
            "epoch": epoch,
            "epoch_val_loss": val_loss,
            "epoch_val_loss_plot": wandb.Image(f'epoch_{epoch+1}_val_loss.png')
        })
        
        print(f"[TIME] Validation stats - "
              f"Avg batch time: {np.mean(val_batch_times):.4f}s, "
              f"Avg data loading time: {np.mean(val_data_loading_times):.4f}s, "
              f"Avg to device time: {np.mean(val_to_device_times):.4f}s, "
              f"Avg forward time: {np.mean(val_forward_times):.4f}s")
    else:
        val_loss = float('inf')
        t_end = time.time()
        all_val_losses = []
        print("WARNING: No validation batches were processed in this epoch!")
    
    return val_loss, all_val_losses, t_end - t_start


def evaluate(model, test_loader, criterion, device):
    t_start = time.time()
    
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_outputs = []
    
    test_batch_times = []
    test_data_loading_times = []
    test_to_device_times = []
    test_forward_times = []
    
    with torch.no_grad():
        for i, (images, morphology, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            t_test_batch_start = time.time()
            t_test_data_load_end = time.time()
            test_data_loading_times.append(t_test_data_load_end - t_test_batch_start)
            
            t_test_to_device_start = time.time()
            images = images.to(device, non_blocking=True)
            morphology = morphology.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            t_test_to_device_end = time.time()
            test_to_device_times.append(t_test_to_device_end - t_test_to_device_start)
            
            # Skip singleton batches during testing too
            if images.size(0) < 2:
                print(f"WARNING: Skipping test batch {i} with only {images.size(0)} samples")
                continue
            
            t_test_forward_start = time.time()
            with torch.amp.autocast('cuda'):
                outputs = model(images, morphology)
                loss = criterion(outputs, targets)
            t_test_forward_end = time.time()
            test_forward_times.append(t_test_forward_end - t_test_forward_start)
                
            batch_loss = loss.item()
            test_loss += batch_loss * images.size(0)
            
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
            
            wandb.log({
                "test_batch": i,
                "test_batch_loss": batch_loss
            })
            
            t_test_batch_end = time.time()
            test_batch_times.append(t_test_batch_end - t_test_batch_start)
            
            if i == 0 or (i + 1) % 10 == 0:
                print(f"[TIME] Test batch {i+1} breakdown - "
                      f"Total: {test_batch_times[-1]:.4f}s, "
                      f"Data loading: {test_data_loading_times[-1]:.4f}s, "
                      f"To device: {test_to_device_times[-1]:.4f}s, "
                      f"Forward: {test_forward_times[-1]:.4f}s, "
                      f"Loss: {batch_loss:.6f}")
    
    print(f"[TIME] Test stats - "
          f"Avg batch time: {np.mean(test_batch_times):.4f}s, "
          f"Avg data loading time: {np.mean(test_data_loading_times):.4f}s, "
          f"Avg to device time: {np.mean(test_to_device_times):.4f}s, "
          f"Avg forward time: {np.mean(test_forward_times):.4f}s")
    
    # Check if we have any outputs to process
    if all_targets and all_outputs:
        t_numpy_conversion_start = time.time()
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)
        t_numpy_conversion_end = time.time()
        print(f"[TIME] NumPy conversion: {t_numpy_conversion_end - t_numpy_conversion_start:.4f}s")
        
        test_loss /= len(test_loader.dataset)
        t_end = time.time()
        
        wandb.log({"final_test_loss": test_loss})
    else:
        test_loss = float('inf')
        all_targets = np.array([])
        all_outputs = np.array([])
        t_end = time.time()
        print("WARNING: No test batches produced any outputs!")
    
    return test_loss, all_targets, all_outputs, t_end - t_start


def calculate_metrics(all_targets, all_outputs, biomarkers):
    t_start = time.time()
    

    if all_targets.size == 0 or all_outputs.size == 0:
        print("WARNING: No data available to calculate metrics")
        return pd.DataFrame()
    
    results = {}
    
    for i, biomarker in enumerate(biomarkers):
        t_metric_biomarker_start = time.time()
        
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
        
        wandb.log({
            f"{biomarker}_PearsonR": pearson_r,
            f"{biomarker}_SpearmanR": spearman_r,
            f"{biomarker}_C-index": c_index
        })
        
        t_metric_biomarker_end = time.time()
        if i == 0 or (i + 1) % 5 == 0:
            print(f"[TIME] Metrics calculation for {biomarker}: {t_metric_biomarker_end - t_metric_biomarker_start:.4f}s")
    
    results_df = pd.DataFrame(results).T
    
    t_end = time.time()
    print(f"[TIME] Total metrics calculation: {t_end - t_start:.4f}s")
    
    return results_df


def save_results(results_df, all_outputs, test_df, biomarkers, batch_size, output_dir):
    t_start = time.time()
    
    # Check if we have valid results to save
    if results_df.empty:
        print("WARNING: No results to save")
        return
    
    t_save_metrics_start = time.time()
    results_df.to_csv(os.path.join(output_dir, "biomarker_metrics_cnn.csv"))
    t_save_metrics_end = time.time()
    print(f"[TIME] Metrics saving: {t_save_metrics_end - t_save_metrics_start:.4f}s")
    
    t_predictions_start = time.time()
    # Ensure test_df has enough rows
    if len(test_df) == 0:
        print("WARNING: Empty test dataframe, cannot save predictions")
        return
    
    test_predictions = np.zeros((len(test_df), len(biomarkers)))
    

    if all_outputs.size > 0:
        batch_idx = 0
        for idx, (start_idx, end_idx) in enumerate([(i, min(i + batch_size, len(test_df))) 
                                                for i in range(0, len(test_df), batch_size)]):
            batch_size_actual = end_idx - start_idx
            if batch_idx + batch_size_actual <= all_outputs.shape[0]:
                test_predictions[start_idx:end_idx] = all_outputs[batch_idx:batch_idx+batch_size_actual]
                batch_idx += batch_size_actual
            else:
                print(f"WARNING: Not enough predictions for all test samples. Using zeros for indices {start_idx}:{end_idx}")
        
        for i, biomarker in enumerate(biomarkers):
            test_df[f"{biomarker}_predicted"] = np.expm1(np.clip(test_predictions[:, i], -10, 10))
    else:
        print("WARNING: No outputs available, predictions will be all zeros")
        for i, biomarker in enumerate(biomarkers):
            test_df[f"{biomarker}_predicted"] = 0.0
    
    t_save_predictions_start = time.time()
    test_df.to_csv(os.path.join(output_dir, f"cell_predictions_cnn.csv"), index=False)
    t_save_predictions_end = time.time()
    print(f"[TIME] Predictions saving: {t_save_predictions_end - t_save_predictions_start:.4f}s")
    
    t_end = time.time()
    print(f"[TIME] Total results saving: {t_end - t_start:.4f}s")

    pred_cols = [c for c in test_df.columns if 'predicted' in c or c in biomarkers]
    if pred_cols:
        sample_df = test_df[pred_cols].head(100)  
        wandb.log({"predictions_sample": wandb.Table(dataframe=sample_df)})


def save_model(model, biomarkers, morphology_features, hidden_dim, output_dim, output_dir):
    t_start = time.time()
    
    model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    
    torch.save({
        'model_state_dict': model_state_dict,
        'biomarkers': biomarkers,
        'morphology_features': morphology_features,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim
    }, os.path.join(output_dir, "biomarker_model_cnn.pt"))
    
    t_end = time.time()
    print(f"[TIME] Model saving: {t_end - t_start:.4f}s")


def train_pipeline(data_directory, biomarkers, morphology_features=None, batch_size=3, num_epochs=30, hidden_dim=256):
    t_total_start = time.time()
    
    wandb.init(project="biomarker-prediction", name=f"vit-model-bs{batch_size}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU.")
    
    device = torch.device("cuda")
    scaler = GradScaler(enabled=True)
    
    output_directory = "./results"
    os.makedirs(output_directory, exist_ok=True)
    output_dir = os.path.join(output_directory, "model_results_cnn")
    os.makedirs(output_dir, exist_ok=True)
    
    batch_checkpoints_dir = os.path.join(output_dir, "batch_checkpoints")
    os.makedirs(batch_checkpoints_dir, exist_ok=True)
    
    if morphology_features is None:
        morphology_features = ["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", 
                               "Solidity", "Extent", "Perimeter", "X_centroid", "Y_centroid"]
    
    all_data_files = find_data_files(data_directory)
    
    train_files, val_files, test_files = create_data_splits(all_data_files)
    
    train_dataset, train_loader = create_datasets_and_loaders(train_files, biomarkers, morphology_features, batch_size)
    val_dataset, val_loader = create_datasets_and_loaders(val_files, biomarkers, morphology_features, batch_size)
    test_dataset, test_loader = create_datasets_and_loaders(test_files, biomarkers, morphology_features, batch_size)
    
    if isinstance(train_dataset, torch.utils.data.ConcatDataset):
        if hasattr(train_dataset.datasets[0], 'morphology_features'):
            morphology_dim = len(train_dataset.datasets[0].morphology_features)
        else:
            print("WARNING: Could not determine morphology_dim from dataset, using default")
            morphology_dim = len(morphology_features)
    else:
        morphology_dim = len(train_dataset.morphology_features)
    
    output_dim = len(biomarkers)
    
    print(f"Creating Vision Transformer model with {morphology_dim} morphology features")
    model = HEBiomarkerModel(morphology_dim, hidden_dim, output_dim)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    wandb.watch(model, log="all")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader),
                                             epochs=num_epochs, pct_start=0.3, div_factor=10, final_div_factor=100)
    
    best_val_loss = float('inf')
    
    print(f"Training CNN model for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        
        train_loss, train_losses, train_time = train_epoch(model, train_loader, criterion, optimizer, scheduler, 
                                                        scaler, device, epoch, num_epochs, batch_checkpoints_dir)
        val_loss, val_losses, val_time = validate(model, val_loader, criterion, device, epoch, num_epochs)
        
        t_epoch_end = time.time()
        epoch_time = t_epoch_end - t_epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Epoch time: {timedelta(seconds=epoch_time)}, "
              f"Train time: {timedelta(seconds=train_time)}, "
              f"Val time: {timedelta(seconds=val_time)}")
        
        if train_losses and val_losses:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train')
            plt.title(f'Epoch {epoch+1} Training Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(val_losses, label='Validation')
            plt.title(f'Epoch {epoch+1} Validation Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'epoch_{epoch+1}_combined_loss.png')
            plt.close()
            
            wandb.log({
                "epoch": epoch,
                "epoch_combined_loss_plot": wandb.Image(f'epoch_{epoch+1}_combined_loss.png'),
                "epoch_time": epoch_time
            })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            t_save_start = time.time()
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'biomarkers': biomarkers,
                'morphology_features': morphology_features
            }, os.path.join(output_dir, "best_biomarker_cnn.pt"))
            t_save_end = time.time()
            print(f"[TIME] Checkpoint saving: {t_save_end - t_save_start:.4f}s")
            
            # Log best model to wandb
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch
    
    torch.cuda.empty_cache()
    
    # Test with the best model
    print("Loading best model for testing...")
    try:
        checkpoint = torch.load(os.path.join(output_dir, "best_biomarker_cnn.pt"))
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        print("Evaluating on test set...")
        test_dfs = []
        for csv_file, _, _ in test_files:
            try:
                df = pd.read_csv(csv_file)
                test_dfs.append(df)
            except Exception as e:
                print(f"WARNING: Could not load test CSV file {csv_file}: {e}")
        
        if test_dfs:
            test_df = pd.concat(test_dfs, ignore_index=True)
            test_loss, all_targets, all_outputs, test_time = evaluate(model, test_loader, criterion, device)
            print(f"Test Loss: {test_loss:.4f}")
            
            results_df = calculate_metrics(all_targets, all_outputs, biomarkers)
            save_results(results_df, all_outputs, test_df, biomarkers, batch_size, output_dir)
        else:
            print("WARNING: No test data available")
    except Exception as e:
        print(f"ERROR during testing: {e}")
    
    save_model(model, biomarkers, morphology_features, hidden_dim, output_dim, output_dir)
    
    t_total_end = time.time()
    total_time = t_total_end - t_total_start
    
    print("\n===== TIMING SUMMARY =====")
    print(f"Total pipeline time: {timedelta(seconds=total_time)}")
    print("==========================\n")
    

    wandb.run.summary["training_time"] = total_time
    wandb.finish()
    
    print(f"Results saved to {output_dir}")
    return pd.DataFrame() if 'results_df' not in locals() else results_df


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    data_directory = "../data/data"
    biomarkers = [
        "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", "CD8a",
        "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", "PD-1", "Ki67", "Pan-CK", "SMA"
    ]
    
    morphology_features = [
        "Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", 
        "Solidity", "Extent", "Perimeter", "X_centroid", "Y_centroid"
    ]
    
    batch_size = 1024 
    
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    train_pipeline(data_directory, biomarkers, morphology_features, batch_size=batch_size, num_epochs=1)

# CUDA_VISIBLE_DEVICES=3,4,5,6,7 nohup python vit.py > vit.log 2>&1 &
