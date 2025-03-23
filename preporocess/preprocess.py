import cv2
import numpy as np
import os
import tifffile
from tqdm import tqdm
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
import psutil
import time
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Global flag to use GPU acceleration
USE_GPU = True

def memory_status():
    """Log current memory usage"""
    mem = psutil.virtual_memory()
    logger.info(f"Memory: {mem.used/1024/1024/1024:.1f}GB / {mem.total/1024/1024/1024:.1f}GB ({mem.percent}%)")

def apply_clahe_gpu(img):
    """Apply Contrast Limited Adaptive Histogram Equalization using GPU acceleration."""
    try:
        if len(img.shape) == 3:  # Color image
            # Upload to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            # Convert to YUV on GPU
            gpu_yuv = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_RGB2YUV)
            # Split channels (returns a list of GpuMat)
            channels = cv2.cuda.split(gpu_yuv)
            # Create GPU CLAHE
            clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            channels[0] = clahe.apply(channels[0])
            # Merge channels back
            gpu_merged = cv2.cuda.merge(channels)
            # Convert back to RGB on GPU
            gpu_result = cv2.cuda.cvtColor(gpu_merged, cv2.COLOR_YUV2RGB)
            result = gpu_result.download()
            return result
        else:  # Grayscale image
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)
            clahe = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gpu_result = clahe.apply(gpu_img)
            result = gpu_result.download()
            return result
    except Exception as e:
        logger.error(f"GPU CLAHE failed, falling back to CPU: {e}")
        # Fallback to CPU CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img.shape) == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            return clahe.apply(img)

def apply_clahe(img):
    """Apply CLAHE using GPU if available; otherwise, use CPU."""
    if USE_GPU:
        return apply_clahe_gpu(img)
    else:
        if len(img.shape) == 3:  
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(img)

def warpAffine_gpu(src_img, affine_M, size):
    """Apply affine transformation using GPU acceleration."""
    try:
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(src_img)
        affine_M = affine_M.astype(np.float32)
        gpu_warped = cv2.cuda.warpAffine(gpu_src, affine_M, size, flags=cv2.INTER_LINEAR)
        result = gpu_warped.download()
        return result
    except Exception as e:
        logger.error(f"GPU warpAffine failed, falling back to CPU: {e}")
        return cv2.warpAffine(src_img, affine_M, size, flags=cv2.INTER_LINEAR)

def align_images_sift_ransac(src_img, dst_img):
    """
    Align images using SIFT feature matching with RANSAC for affine transformation.
    GPU acceleration is used for CLAHE and warping.
    
    Args:
        src_img: The image to transform (H&E)
        dst_img: The reference image (IF)
        
    Returns:
        Transformed source image or None if alignment fails.
    """
    try:
        # Convert images to grayscale if needed
        if len(src_img.shape) == 3:
            src_gray = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
        else:
            src_gray = src_img
        if len(dst_img.shape) == 3:
            dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_RGB2GRAY)
        else:
            dst_gray = dst_img
        
        # Apply CLAHE using GPU acceleration if enabled
        src_gray = apply_clahe(src_gray)
        dst_gray = apply_clahe(dst_gray)
        
        # Detect features using CPU SIFT (GPU SIFT is not available in CUDA)
        sift = cv2.SIFT_create(nfeatures=5000)
        keypoints1, descriptors1 = sift.detectAndCompute(src_gray, None)
        keypoints2, descriptors2 = sift.detectAndCompute(dst_gray, None)
        
        if len(keypoints1) < 4 or len(keypoints2) < 4:
            logger.warning("Not enough keypoints found for alignment")
            return None
        
        # Match descriptors using FLANN-based matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            logger.warning(f"Not enough good matches found: {len(good_matches)}")
            return None
        
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            logger.warning("Failed to find homography matrix")
            return None
        
        # Construct affine matrix (2x3) from homography for warpAffine
        affine_M = np.array([
            [M[0, 0], M[0, 1], M[0, 2]],
            [M[1, 0], M[1, 1], M[1, 2]]
        ])
        
        h, w = dst_img.shape[:2]
        aligned_img = warpAffine_gpu(src_img, affine_M, (w, h)) if USE_GPU else cv2.warpAffine(src_img, affine_M, (w, h), flags=cv2.INTER_LINEAR)
        return aligned_img
    
    except Exception as e:
        logger.error(f"Error in alignment: {e}")
        return None

def get_if_reference_image(if_path, channel_idx=0):
    try:
        logger.info("Extracting reference IF image from channel 0")
        ref_image = tifffile.imread(if_path, key=range(channel_idx, channel_idx+1))[0]
        if ref_image is None:
            raise ValueError("Failed to extract reference image from IF file")
        if ref_image.dtype == np.uint16:
            ref_image = cv2.normalize(ref_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif ref_image.dtype != np.uint8:
            ref_image = (255 * (ref_image - ref_image.min()) / (ref_image.max() - ref_image.min() + 1e-10)).astype(np.uint8)
        return ref_image
    except Exception as e:
        logger.error(f"Error extracting reference image: {e}")
        return None

def process_if_chunk(args):
    """Process a chunk of the IF image for a specific channel."""
    channel_idx, y_start, y_end, x_start, x_end, if_path = args
    try:
        chunk = tifffile.imread(if_path, key=channel_idx, series=0, 
                                  aszarr=True, out_shape=(y_end-y_start, x_end-x_start), 
                                  out=None, level=0, 
                                  region=(0, y_start, x_start, 1, y_end-y_start, x_end-x_start))
        if chunk is None:
            return (y_start, y_end, x_start, x_end, np.zeros((y_end-y_start, x_end-x_start), dtype=np.uint8))
        if chunk.dtype == np.uint16:
            chunk_8bit = cv2.normalize(chunk, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif chunk.dtype == np.uint8:
            chunk_8bit = chunk
        else:
            chunk_8bit = (255 * (chunk - chunk.min()) / (chunk.max() - chunk.min() + 1e-10)).astype(np.uint8)
        
        # Apply CLAHE (GPU-accelerated if enabled)
        chunk_8bit = apply_clahe(chunk_8bit)
        return (y_start, y_end, x_start, x_end, chunk_8bit)
    
    except Exception as e:
        logger.error(f"Error processing IF chunk at channel {channel_idx}, position ({y_start},{x_start}): {e}")
        return (y_start, y_end, x_start, x_end, np.zeros((y_end-y_start, x_end-x_start), dtype=np.uint8))

def process_channel(channel_idx, if_height, if_width, chunk_size, if_path, output_dir, marker_name):
    """Process an entire IF channel and save it as a separate file."""
    logger.info(f"Processing channel {channel_idx} - {marker_name}...")
    channels_dir = os.path.join(output_dir, "channels")
    os.makedirs(channels_dir, exist_ok=True)
    channel_image = np.zeros((if_height, if_width), dtype=np.uint8)
    
    chunk_coords = []
    for y_start in range(0, if_height, chunk_size):
        y_end = min(y_start + chunk_size, if_height)
        for x_start in range(0, if_width, chunk_size):
            x_end = min(x_start + chunk_size, if_width)
            chunk_coords.append((channel_idx, y_start, y_end, x_start, x_end, if_path))
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_if_chunk, coords) for coords in chunk_coords]
        for future in tqdm(as_completed(futures), total=len(futures), 
                           desc=f"Channel {channel_idx} - {marker_name}"):
            result = future.result()
            if result:
                y_start, y_end, x_start, x_end, chunk_data = result
                channel_image[y_start:y_end, x_start:x_end] = chunk_data
    
    channel_file = os.path.join(channels_dir, f"{channel_idx:02d}_{marker_name}.tif")
    tifffile.imwrite(channel_file, channel_image, compress=6)
    return channel_image

def preprocess_he_if_images(he_path, if_path, output_dir, biomarker_names, chunk_size=10000):
    """Main preprocessing function with optimizations for large images using GPU acceleration where applicable."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    memory_status()
    
    # Predefined image dimensions (adjust as necessary)
    he_height, he_width, channels = 57360, 78417, 3
    if_height, if_width = he_height, he_width  
    if_channels = len(biomarker_names)
    
    logger.info(f"Using predefined image dimensions: {he_height}x{he_width}, {if_channels} channels")
    logger.info(f"Using chunk size: {chunk_size}x{chunk_size}")
    logger.info("Reading reference IF image for alignment...")
    if_ref_image = get_if_reference_image(if_path)
    if if_ref_image is None:
        logger.error("Failed to read reference IF image")
        return False
    
    logger.info("Reading full resolution H&E image...")
    he_image = tifffile.imread(he_path, key=0)
    
    logger.info("Enhancing H&E image...")
    he_image_enhanced = apply_clahe(he_image)
    
    logger.info("Performing alignment on full resolution...")
    aligned_he = align_images_sift_ransac(he_image_enhanced, if_ref_image)
    if aligned_he is None:
        logger.error("Alignment failed. Using original H&E image resized to IF dimensions")
        aligned_he = cv2.resize(he_image, (if_width, if_height))
    
    aligned_he_path = os.path.join(output_dir, "aligned_he.tif")
    logger.info(f"Saving aligned H&E image to {aligned_he_path}")
    tifffile.imwrite(aligned_he_path, aligned_he, compress=6)
    
    del he_image, he_image_enhanced, if_ref_image
    gc.collect()
    memory_status()
    
    logger.info("Processing IF channels...")
    max_workers = min(if_channels, multiprocessing.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batch_size = 4
        for batch_start in range(0, if_channels, batch_size):
            batch_end = min(batch_start + batch_size, if_channels)
            logger.info(f"Processing channel batch {batch_start+1}-{batch_end} of {if_channels}")
            futures = []
            for i in range(batch_start, batch_end):
                if i >= len(biomarker_names):
                    continue
                marker_name = biomarker_names[i]
                futures.append(executor.submit(
                    process_channel, 
                    i, if_height, if_width, chunk_size, if_path, output_dir, marker_name
                ))
            for future in as_completed(futures):
                _ = future.result()
            gc.collect()
            memory_status()
    
    logger.info("Creating merged RGB visualization...")
    try:
        vis_indices = [0, 2, 6]
        vis_channels = []
        channels_dir = os.path.join(output_dir, "channels")
        for i in vis_indices:
            if i >= len(biomarker_names):
                continue
            channel_file = os.path.join(channels_dir, f"{i:02d}_{biomarker_names[i]}.tif")
            if os.path.exists(channel_file):
                ch = tifffile.imread(channel_file)
                vis_channels.append(ch)
        if len(vis_channels) >= 3:
            rgb_vis = np.zeros((if_height, if_width, 3), dtype=np.uint8)
            rgb_vis[:, :, 0] = vis_channels[0]
            rgb_vis[:, :, 1] = vis_channels[1]
            rgb_vis[:, :, 2] = vis_channels[2]
            vis_path = os.path.join(output_dir, "merged_visualization.tif")
            tifffile.imwrite(vis_path, rgb_vis, compress=6)
            logger.info(f"RGB visualization saved to {vis_path}")
        else:
            logger.warning("Not enough channels for RGB visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Processing complete in {elapsed_time/60:.1f} minutes!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess H&E and IF images with parallel processing and GPU acceleration')
    parser.add_argument('--he', required=True, help='Path to H&E image')
    parser.add_argument('--If', required=True, help='Path to IF image')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=10000, help='Chunk size for processing')
    parser.add_argument('--biomarkers', default=None, help='Comma-separated list of biomarker names')
    args = parser.parse_args()
    
    if args.biomarkers:
        biomarker_names = args.biomarkers.split(',')
    else:
        biomarker_names = [
            "Hoechst", "AF1", "CD31", "CD45", "CD68", "Argo550", "CD4", "FOXP3", 
            "CD8a", "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-cadherin", 
            "PD-1", "Ki67", "Pan-CK", "SMA"
        ]
    
    preprocess_he_if_images(args.he, args.If, args.out, biomarker_names, args.chunk_size)
