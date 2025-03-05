import numpy as np
import tifffile as tiff
import time
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index
import gc
import os

def process_biomarker(i, biomarker_name, hefile_path, iffile_path, sample_size=500_000):
    """Process a single biomarker following the paper's approach."""
    start_time = time.time()
    print(f"Processing {biomarker_name}...")
    
    print("  Loading H&E image...")
    HE_image = tiff.imread(hefile_path)
    
    HE_mean = np.mean(HE_image, axis=2).astype(np.float32) / 255.0
    HE_expression = 1.0 - HE_mean  
    
    HE_flat = HE_expression.ravel()
    
    del HE_image
    del HE_expression
    gc.collect()

    print(f"  Loading biomarker {i}...")
    biomarker = tiff.imread(iffile_path, key=i)
    biomarker_flat = biomarker.astype(np.float32).ravel() / 65535.0
    
    del biomarker
    gc.collect()
    
    if len(HE_flat) > sample_size:
        np.random.seed(42)
        indices = np.random.choice(len(HE_flat), size=sample_size, replace=False)
        HE_sample = HE_flat[indices]
        biomarker_sample = biomarker_flat[indices]
    else:
        HE_sample = HE_flat
        biomarker_sample = biomarker_flat
    
    del HE_flat
    del biomarker_flat
    gc.collect()
    
    pearson_r, _ = pearsonr(HE_sample, biomarker_sample)
    spearman_r, _ = spearmanr(HE_sample, biomarker_sample)
    c_index = concordance_index(HE_sample, biomarker_sample)
    
    processing_time = time.time() - start_time
    print(f"  Completed {biomarker_name} in {processing_time:.2f} seconds")
    print(f"  Results: Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f}, C-index={c_index:.3f}")
    
    return {
        "biomarker": biomarker_name,
        "Pearson R": pearson_r,
        "Spearman R": spearman_r,
        "C-index": c_index,
        "processing_time": processing_time
    }

def main():
    start_total = time.time()
    
    hefile_path = "../data/data/CRC01/18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif"
    iffile_path = "../data/data/CRC01/P37_S29_A24_C59kX_E15_20220106_014304_946511-zlib.ome.tiff"
    
    biomarker_names = [
        "Hoechst (Nucleus Staining)", "AF1", "CD31 (Endothelial Cells)", "CD45 (Immune Cells)", 
        "CD68 (Macrophages)", "Argo550", "CD4 (Helper T Cells)", "FOXP3 (Regulatory T Cells)",
        "CD8a (Cytotoxic T Cells)", "CD45RO (Memory T Cells)", "CD20 (B Cells)", "PD-L1 (Immune Checkpoint)",
        "CD3e (General T Cells)", "CD163 (M2 Macrophages)", "E-cadherin (Epithelial Cells)", 
        "PD-1 (Immune Checkpoint)", "Ki67 (Proliferation Marker)", "Pan-CK (Tumor Cells)", "SMA (Fibroblasts)"
    ]

    results = {}
    
    for i, biomarker_name in enumerate(biomarker_names):
        result = process_biomarker(i, biomarker_name, hefile_path, iffile_path)
        results[biomarker_name] = result
    
    print("\nFinal Results Summary:")
    print("=" * 80)
    print(f"{'Biomarker':<30} {'Pearson R':>10} {'Spearman R':>12} {'C-index':>10}")
    print("-" * 80)
    
    for biomarker_name in biomarker_names:
        scores = results[biomarker_name]
        print(f"{biomarker_name:<30} {scores['Pearson R']:>10.3f} {scores['Spearman R']:>12.3f} "
              f"{scores['C-index']:>10.3f}")
    
    avg_pearson = np.mean([scores['Pearson R'] for scores in results.values()])
    avg_spearman = np.mean([scores['Spearman R'] for scores in results.values()])
    avg_cindex = np.mean([scores['C-index'] for scores in results.values()])
    
    print("\nAverage metrics across all biomarkers:")
    print(f"Average Pearson R: {avg_pearson:.3f}")
    print(f"Average Spearman R: {avg_spearman:.3f}")
    print(f"Average C-index: {avg_cindex:.3f}")
    
    print(f"\nTotal execution time: {time.time() - start_total:.2f} seconds")

if __name__ == "__main__":
    main()