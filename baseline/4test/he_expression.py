import numpy as np
import tifffile as tiff
import time
import zarr
from scipy.stats import spearmanr, pearsonr
from lifelines.utils import concordance_index
import gc
import os
import psutil
from joblib import Parallel, delayed

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Convert to GB

def convert_tiff_to_zarr(tiff_path, zarr_path):
    """Convert a large TIFF file to Zarr format for faster random access."""
    if not os.path.exists(zarr_path):
        print(f"Converting {tiff_path} to {zarr_path} for faster access...")
        img = tiff.imread(tiff_path)
        zarr.save(zarr_path, img)
        print(f"Conversion complete: {zarr_path}")
    return zarr.load(zarr_path)

def process_chunk_means(he_chunk, bio_chunk):
    return np.sum(he_chunk), np.sum(bio_chunk), he_chunk.size

def process_chunk_cov(he_chunk, bio_chunk, he_mean, bio_mean):
    he_diff = he_chunk - he_mean
    bio_diff = bio_chunk - bio_mean
    return np.sum(he_diff * he_diff), np.sum(bio_diff * bio_diff), np.sum(he_diff * bio_diff), he_chunk.size

def process_chunk_corr(he_chunk, bio_chunk):
    if he_chunk.size > 1_000_000:
        indices = np.random.choice(he_chunk.size, size=1_000_000, replace=False)
        he_sample = he_chunk[indices]
        bio_sample = bio_chunk[indices]
    else:
        he_sample = he_chunk
        bio_sample = bio_chunk
    pearson_r, _ = pearsonr(he_sample, bio_sample)
    spearman_r, _ = spearmanr(he_sample, bio_sample)
    c_index = concordance_index(he_sample, bio_sample)
    return pearson_r, spearman_r, c_index, he_chunk.size

def parallel_process_biomarker(i, biomarker_name, hefile_path, iffile_path, num_processes):
    start_time = time.time()
    print(f"Processing {biomarker_name} using {num_processes} processes...")
    print(f"Memory before loading: {get_memory_usage():.2f} GB")

    # --- Load H&E Image (Convert to Zarr for faster access) ---
    HE_image = convert_tiff_to_zarr(hefile_path, "HE_image.zarr")
    HE_mean = np.mean(HE_image, axis=2, dtype=np.float32) / 255.0
    HE_expression = 1.0 - HE_mean
    del HE_image, HE_mean
    gc.collect()
    print(f"Memory after H&E processing: {get_memory_usage():.2f} GB")

    # --- Load Biomarker Image (Zarr conversion) ---
    biomarker = convert_tiff_to_zarr(iffile_path, f"biomarker_{i}.zarr")
    biomarker_float = biomarker.astype(np.float32) / 65535.0
    del biomarker
    gc.collect()
    print(f"Memory after biomarker loading: {get_memory_usage():.2f} GB")

    # --- Ensure Shape Consistency ---
    if HE_expression.shape != biomarker_float.shape:
        min_rows, min_cols = min(HE_expression.shape[0], biomarker_float.shape[0]), min(HE_expression.shape[1], biomarker_float.shape[1])
        HE_expression, biomarker_float = HE_expression[:min_rows, :min_cols], biomarker_float[:min_rows, :min_cols]

    total_pixels = HE_expression.size
    print(f"Total pixels to process: {total_pixels:,}")

    # --- Flatten the Arrays ---
    HE_flat, biomarker_flat = HE_expression.ravel(), biomarker_float.ravel()
    del HE_expression, biomarker_float
    gc.collect()
    print(f"Memory after flattening: {get_memory_usage():.2f} GB")

    # --- Set Chunk Size (50M Pixels for Efficient Parallelization) ---
    chunk_size = 50_000_000
    num_chunks = (total_pixels + chunk_size - 1) // chunk_size
    print(f"Using {num_chunks} chunks with ~{chunk_size:,} pixels per chunk")

    # --- Compute Means in Parallel ---
    chunk_means_results = Parallel(n_jobs=num_processes)(
        delayed(process_chunk_means)(
            HE_flat[j * chunk_size: min((j + 1) * chunk_size, total_pixels)],
            biomarker_flat[j * chunk_size: min((j + 1) * chunk_size, total_pixels)]
        ) for j in range(num_chunks)
    )
    total_he_sum, total_bio_sum, total_count = map(sum, zip(*chunk_means_results))
    he_mean_val, bio_mean_val = total_he_sum / total_count, total_bio_sum / total_count

    # --- Compute Covariance in Parallel ---
    chunk_cov_results = Parallel(n_jobs=num_processes)(
        delayed(process_chunk_cov)(
            HE_flat[j * chunk_size: min((j + 1) * chunk_size, total_pixels)],
            biomarker_flat[j * chunk_size: min((j + 1) * chunk_size, total_pixels)],
            he_mean_val, bio_mean_val
        ) for j in range(num_chunks)
    )
    total_he_var_sum, total_bio_var_sum, total_cov_sum, total_count_cov = map(sum, zip(*chunk_cov_results))
    he_std, bio_std = np.sqrt(total_he_var_sum / total_count_cov), np.sqrt(total_bio_var_sum / total_count_cov)
    covariance, pearson_r = total_cov_sum / total_count_cov, covariance / (he_std * bio_std)

    # --- Compute Correlations (Sampling for Speed) ---
    if total_pixels > 1_000_000_000:
        num_corr_chunks = min(num_processes, total_pixels // 1_000_000)
        np.random.seed(42)
        corr_results = Parallel(n_jobs=num_processes)(
            delayed(process_chunk_corr)(
                np.random.choice(HE_flat, size=1_000_000, replace=False),
                np.random.choice(biomarker_flat, size=1_000_000, replace=False)
            ) for _ in range(num_corr_chunks)
        )
        total_count_corr = sum(res[3] for res in corr_results)
        avg_spearman = sum(res[1] * res[3] for res in corr_results) / total_count_corr
        avg_c_index = sum(res[2] * res[3] for res in corr_results) / total_count_corr
    else:
        avg_spearman, _ = spearmanr(HE_flat, biomarker_flat)
        avg_c_index = concordance_index(HE_flat, biomarker_flat)

    del HE_flat, biomarker_flat
    gc.collect()
    processing_time = time.time() - start_time
    print(f"Completed {biomarker_name} in {processing_time:.2f} seconds")
    return {"biomarker": biomarker_name, "Pearson R": pearson_r, "Spearman R": avg_spearman, "C-index": avg_c_index, "processing_time": processing_time}

def main():
    hefile_path, iffile_path = "HE_image.tif", "IF_image.tif"
    biomarker_names = ["Hoechst", "AF1", "CD31", "CD45", "CD68"]
    num_cpus = os.cpu_count()
    optimal_processes = min(64, num_cpus // 2)
    print(f"Using {optimal_processes} parallel processes")
    results = {b: parallel_process_biomarker(i, b, hefile_path, iffile_path, optimal_processes) for i, b in enumerate(biomarker_names)}
    print("\nFinal Results:", results)

if __name__ == "__main__":
    main()
