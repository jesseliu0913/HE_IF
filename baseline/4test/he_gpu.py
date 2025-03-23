#!/usr/bin/env python3
"""
Optimized version of the ROSIE H&E baseline for very large OME-TIFF files
Uses multiprocessing for faster execution while maintaining memory efficiency
"""

import os
import sys
import numpy as np
import warnings
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import time

# Try imports with helpful error messages
try:
    import tifffile
except ImportError:
    print("Error: tifffile package not found. Install with: pip install tifffile")
    sys.exit(1)

def get_tiff_info(tiff_path):
    """Get basic information about a TIFF file without loading it all"""
    try:
        with tifffile.TiffFile(tiff_path) as tif:
            # Get the first page to extract metadata
            page = tif.pages[0]
            shape = page.shape
            dtype = page.dtype
            
            # Check if it's a multi-page TIFF
            num_pages = len(tif.pages)
            
            return {
                'shape': shape,
                'dtype': dtype,
                'num_pages': num_pages
            }
    except Exception as e:
        print(f"Error examining TIFF file: {e}")
        return None

def process_block(block_info, he_path, if_path, threshold=50):
    """
    Process a single block of the image
    
    Args:
        block_info: Tuple of (y, y_end, x, x_end)
        he_path: Path to HE TIFF file
        if_path: Path to IF TIFF file
        threshold: Threshold for H&E expression baseline
        
    Returns:
        Dict with correlation results for this block
    """
    y, y_end, x, x_end = block_info
    
    # Keep trying with progressively smaller blocks if memory error occurs
    for scale_down in [1, 2, 4]:
        try:
            # Adjust block size if needed
            if scale_down > 1:
                y_adj = y
                y_end_adj = y + min((y_end - y) // scale_down, 32)  # Much smaller sub-block
                x_adj = x
                x_end_adj = x + min((x_end - x) // scale_down, 32)
            else:
                y_adj, y_end_adj, x_adj, x_end_adj = y, y_end, x, x_end
            
            # Open H&E file and read only the block we need
            with tifffile.TiffFile(he_path) as he_tif:
                he_block = he_tif.pages[0].asarray()[y_adj:y_end_adj, x_adj:x_end_adj]
            
            # Apply H&E baseline
            if len(he_block.shape) == 3 and he_block.shape[2] == 3:  # RGB
                he_mean = np.mean(he_block, axis=2)
            else:
                he_mean = he_block
            
            mask = (he_mean > threshold).astype(np.float32)
            prediction = he_mean * mask
            
            # Use a minimal memory approach by processing one IF page at a time
            all_pearson_by_page = []
            all_spearman_by_page = []
            
            with tifffile.TiffFile(if_path) as if_tif:
                num_pages = len(if_tif.pages)
                
                # Process each IF page (biomarker) separately to save memory
                for page_idx in range(num_pages):
                    # Load just this small block from the IF page
                    if_page_block = if_tif.pages[page_idx].asarray()[y_adj:y_end_adj, x_adj:x_end_adj]
                    
                    # Calculate correlations for this page/biomarker
                    pred_flat = prediction.flatten()
                    gt_flat = if_page_block.flatten()
                    
                    # Calculate correlations if we have enough valid data points
                    valid_indices = ~np.isnan(gt_flat) & ~np.isnan(pred_flat)
                    if np.sum(valid_indices) > 10:
                        try:
                            # Pearson correlation
                            pearson = np.corrcoef(gt_flat[valid_indices], pred_flat[valid_indices])[0, 1]
                            
                            # Spearman correlation
                            gt_rank = np.argsort(np.argsort(gt_flat[valid_indices]))
                            pred_rank = np.argsort(np.argsort(pred_flat[valid_indices]))
                            spearman = np.corrcoef(gt_rank, pred_rank)[0, 1]
                            
                            all_pearson_by_page.append((pearson, np.sum(valid_indices)))
                            all_spearman_by_page.append((spearman, np.sum(valid_indices)))
                        except:
                            # Skip this page if correlation calculation fails
                            pass
                    
                    # Explicitly free memory
                    del if_page_block, gt_flat
            
            # Free memory
            del he_block, he_mean, mask, prediction, pred_flat
            
            # Return results
            return {
                'pearson': all_pearson_by_page,
                'spearman': all_spearman_by_page,
                'block_coords': (y, y_end, x, x_end)
            }
            
        except Exception as e:
            # If memory error or other error, try with a smaller block
            if scale_down == 4:  # Last attempt failed
                print(f"Error processing block {(y,y_end,x,x_end)}: {e}")
                return {
                    'pearson': [],
                    'spearman': [],
                    'block_coords': (y, y_end, x, x_end),
                    'error': str(e)
                }

def combine_results(all_results, num_biomarkers):
    """Combine results from all processed blocks"""
    pearson_by_biomarker = [[] for _ in range(num_biomarkers)]
    spearman_by_biomarker = [[] for _ in range(num_biomarkers)]
    
    # Collect results from each block
    for result in all_results:
        if 'error' in result:
            continue
            
        pearson_list = result['pearson']
        spearman_list = result['spearman']
        
        # Add results to appropriate biomarker list
        for i, (pearson_result, spearman_result) in enumerate(zip(pearson_list, spearman_list)):
            if i < num_biomarkers:  # Guard against index errors
                pearson_by_biomarker[i].append(pearson_result)
                spearman_by_biomarker[i].append(spearman_result)
    
    # Calculate weighted average for each biomarker
    pearson_results = []
    spearman_results = []
    
    for bio_idx in range(num_biomarkers):
        # Calculate weighted Pearson average
        if pearson_by_biomarker[bio_idx]:
            values, weights = zip(*pearson_by_biomarker[bio_idx])
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_pearson = sum(v * w for v, w in zip(values, weights)) / total_weight
                pearson_results.append(weighted_pearson)
        
        # Calculate weighted Spearman average
        if spearman_by_biomarker[bio_idx]:
            values, weights = zip(*spearman_by_biomarker[bio_idx])
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_spearman = sum(v * w for v, w in zip(values, weights)) / total_weight
                spearman_results.append(weighted_spearman)
    
    # Calculate overall averages
    avg_pearson = np.mean(pearson_results) if pearson_results else np.nan
    avg_spearman = np.mean(spearman_results) if spearman_results else np.nan
    
    return {
        'pearson_r': avg_pearson,
        'spearman_r': avg_spearman,
        'pearson_by_biomarker': pearson_results,
        'spearman_by_biomarker': spearman_results
    }

def process_file_parallel(he_path, if_path, output_dir, block_size=128, threshold=50, processes=None):
    """
    Process large OME-TIFF files in parallel for speed while maintaining memory efficiency
    
    Args:
        he_path: Path to HE TIFF file
        if_path: Path to IF TIFF file
        output_dir: Directory to save results
        block_size: Size of blocks to process in parallel
        threshold: Threshold for H&E expression baseline
        processes: Number of parallel processes (None = auto)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get file info
    print(f"Examining files...")
    he_info = get_tiff_info(he_path)
    if_info = get_tiff_info(if_path)
    
    if not he_info or not if_info:
        print("Could not get file information. Exiting.")
        return
    
    print(f"H&E file: shape={he_info['shape']}, dtype={he_info['dtype']}, pages={he_info['num_pages']}")
    print(f"IF file: shape={if_info['shape']}, dtype={if_info['dtype']}, pages={if_info['num_pages']}")
    
    # Determine file sizes
    try:
        he_bytes = np.prod(he_info['shape']) * np.dtype(he_info['dtype']).itemsize
        if_bytes = np.prod(if_info['shape']) * np.dtype(if_info['dtype']).itemsize * if_info['num_pages']
        
        print(f"H&E file size: {he_bytes / (1024**3):.2f} GB")
        print(f"IF file size: {if_bytes / (1024**3):.2f} GB")
        
        if (he_bytes + if_bytes) > 32 * (1024**3):  # Very large files (>32GB)
            print("Files are extremely large, using extra-small blocks")
            block_size = min(block_size, 64)  # Use smaller blocks
    except:
        print("Could not determine file sizes, using cautious settings")
        block_size = 64
    
    # Get dimensions (handle RGB case)
    if len(he_info['shape']) == 3:
        height, width = he_info['shape'][:2]
    else:
        height, width = he_info['shape']
    
    # Create list of blocks to process
    blocks = []
    for y in range(0, height, block_size):
        y_end = min(y + block_size, height)
        for x in range(0, width, block_size):
            x_end = min(x + block_size, width)
            blocks.append((y, y_end, x, x_end))
    
    print(f"Dividing image into {len(blocks)} blocks of approx. {block_size}x{block_size} pixels")
    
    # Determine number of processes to use
    if processes is None:
        # Use fewer processes if files are very large to avoid memory issues
        if (he_bytes + if_bytes) > 64 * (1024**3):  # >64GB files
            processes = max(1, min(4, mp.cpu_count() // 2))
        else:
            processes = max(1, min(mp.cpu_count() - 1, 8))
    
    print(f"Using {processes} parallel processes")
    
    # Process blocks in parallel
    results = []
    process_func = partial(process_block, he_path=he_path, if_path=if_path, threshold=threshold)
    
    if processes > 1:
        # Use multiprocessing for parallel execution
        # Process blocks in batches to avoid excessive memory usage
        batch_size = min(128, max(1, len(blocks) // (processes * 4)))
        
        processed_blocks = 0
        total_blocks = len(blocks)
        
        with mp.Pool(processes=processes) as pool:
            for i in range(0, len(blocks), batch_size):
                batch = blocks[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(blocks)+batch_size-1)//batch_size} ({len(batch)} blocks)")
                
                batch_results = list(tqdm(
                    pool.imap(process_func, batch),
                    desc="Processing blocks",
                    total=len(batch)
                ))
                
                results.extend(batch_results)
                processed_blocks += len(batch)
                print(f"Progress: {processed_blocks}/{total_blocks} blocks ({processed_blocks/total_blocks*100:.1f}%)")
    else:
        # Use single process execution
        for block in tqdm(blocks, desc="Processing blocks"):
            result = process_func(block)
            results.append(result)
    
    # Combine results
    print("Combining results from all blocks...")
    num_biomarkers = if_info['num_pages']
    combined_results = combine_results(results, num_biomarkers)
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.npz')
    np.savez(results_path, 
            pearson_r=combined_results['pearson_r'],
            spearman_r=combined_results['spearman_r'],
            pearson_by_biomarker=combined_results['pearson_by_biomarker'],
            spearman_by_biomarker=combined_results['spearman_by_biomarker'])
    
    print(f"Results saved to {results_path}")
    print(f"Average Pearson correlation: {combined_results['pearson_r']:.4f}")
    print(f"Average Spearman correlation: {combined_results['spearman_r']:.4f}")
    
    # Report biomarker-specific results
    for i, (pearson, spearman) in enumerate(zip(
        combined_results['pearson_by_biomarker'], 
        combined_results['spearman_by_biomarker']
    )):
        print(f"Biomarker {i+1}: Pearson={pearson:.4f}, Spearman={spearman:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Optimized ROSIE H&E Baseline')
    parser.add_argument('--he_path', type=str, required=True, help='Path to HE TIFF file')
    parser.add_argument('--if_path', type=str, required=True, help='Path to IF TIFF file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--block_size', type=int, default=128, help='Block size for processing')
    parser.add_argument('--threshold', type=int, default=50, help='Threshold for H&E expression baseline')
    parser.add_argument('--processes', type=int, default=None, help='Number of parallel processes (default: auto)')
    args = parser.parse_args()
    
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"Number of CPU cores: {mp.cpu_count()}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process files
    start_time = time.time()
    process_file_parallel(
        args.he_path, args.if_path, args.output_dir, 
        args.block_size, args.threshold, args.processes
    )
    end_time = time.time()
    
    # Report execution time
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    # For command line execution
    if len(sys.argv) == 1:
        # Example for testing
        sys.argv = [
            'optimized_rosie.py',
            '--he_path', '../data/data/CRC01/18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif',
            '--if_path', '../data/data/CRC01/P37_S29_A24_C59kX_E15_20220106_014304_946511-zlib.ome.tiff',
            '--output_dir', './results',
            '--block_size', '64',  # Smaller blocks for very large files
            '--processes', '4'     # Control parallelism explicitly
        ]
    
    main()

    # python he_gpu.py --he_path "../data/data/CRC01/18459_LSP10353_US_SCAN_OR_001__093059-registered.ome.tif" --if_path "../data/data/CRC01/P37_S29_A24_C59kX_E15_20220106_014304_946511-zlib.ome.tiff" --output_dir "./results" --block_size 64 --processes 4