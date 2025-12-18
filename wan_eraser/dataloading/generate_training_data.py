"""
Pre-generate Training Data for Video Inpainting

This script reads an input CSV with video paths, generates masks with motion trajectories,
extracts reference images, and saves everything to disk. It also creates a new CSV file
that can be used with InpaintingDatasetFromCSV.

Input CSV format:
    video_path: path to video
    caption/prompt: text caption

Output:
    - mask_video: per-frame masks as video (.mp4)
    - ref_image: foreground reference image (.png)
    - New CSV with all paths

Usage:
    python generate_training_data.py \
        --input_csv /path/to/input.csv \
        --output_dir /path/to/output \
        --data_root / \
        --nframes 81 \
        --num_workers 8
"""

import os
import csv
import random
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image
from decord import VideoReader
from moviepy import ImageSequenceClip

import sys
sys.path.append("/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/utils")
from saber_mask import generate_mask


# ============================================================================
# Mask Motion Trajectory Generation
# ============================================================================

def save_mask_image(mask: np.ndarray, output_path: str):
    """Save mask as grayscale image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, (mask * 255).astype(np.uint8))


def save_mask_video(masks: List[np.ndarray], output_path: str, fps: int = 24):
    """Save masks as grayscale video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to RGB for moviepy (grayscale repeated 3 times)
    mask_frames = []
    for m in masks:
        mask_rgb = np.stack([m * 255, m * 255, m * 255], axis=-1).astype(np.uint8)
        mask_frames.append(mask_rgb)
    
    clip = ImageSequenceClip(mask_frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", bitrate="5M", logger=None)


def process_single_video(
    row: dict,
    data_root: str,
    output_dir: str,
    nframes: int,
    mask_area_ratio_range: Tuple[float, float] = (0.15, 0.40),
    mask_shape_types: List[str] = None,
) -> Optional[dict]:
    """
    Process a single video: generate static mask and ref_image.
    Returns new row dict with paths, or None on failure.
    """
    if mask_shape_types is None:
        mask_shape_types = ["ellipse", "superellipse", "concave_polygon", "centered_rectangle"]
    
    vid_path = os.path.join(data_root, row.get('video_path', row.get('vid_path', '')))
    caption = row.get('prompt', row.get('caption', ''))
    
    try:
        video_reader = VideoReader(vid_path)
        
        # Handle short videos
        if len(video_reader) < nframes:
            repeat_num = math.ceil(nframes / len(video_reader))
            temp_list = list(range(len(video_reader))) + list(range(len(video_reader)))[::-1][1:-1]
            all_frames = temp_list * repeat_num
        else:
            all_frames = list(range(len(video_reader)))
        
        # Use start of video for consistency
        frame_indices = all_frames[:nframes]
        
        if len(frame_indices) < nframes:
            return None
        
        video = video_reader.get_batch(frame_indices).asnumpy()
        
    except Exception as e:
        print(f"Error loading video {vid_path}: {e}")
        return None
    
    height_v, width_v, _ = video[0].shape
    
    # Generate static mask (same for all frames)
    mask = generate_mask(
        h=height_v,
        w=width_v,
        area_ratio_range=mask_area_ratio_range,
        shape_types=mask_shape_types,
    )
    
    # Extract reference image (first frame foreground)
    first_frame = video[0].copy()
    first_frame[mask < 0.5] = 0  # Keep only mask==1 region
    
    # Create output paths
    video_name = Path(vid_path).stem
    sample_dir = Path(output_dir) / video_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    mask_image_path = str(sample_dir / "mask.png")
    ref_image_path = str(sample_dir / "ref_image.png")
    
    # Save static mask image
    save_mask_image(mask, mask_image_path)
    
    # Save reference image
    cv2.imwrite(ref_image_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    # Return new row with relative paths
    rel_mask_path = os.path.relpath(mask_image_path, output_dir)
    rel_ref_path = os.path.relpath(ref_image_path, output_dir)
    
    return {
        'video_path': row.get('video_path', row.get('vid_path', '')),
        'mask_path': rel_mask_path,
        'ref_image_path': rel_ref_path,
        'caption': caption,
    }


def generate_training_data(
    input_csv: str,
    output_dir: str,
    data_root: str,
    nframes: int = 81,
    num_workers: int = 8,
    mask_area_ratio_range: Tuple[float, float] = (0.15, 0.40),
):
    """
    Generate training data for all videos in CSV.
    """
    # Load input CSV
    rows = []
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Loaded {len(rows)} videos from {input_csv}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process videos
    results = []
    
    if num_workers <= 1:
        # Sequential processing
        for row in tqdm(rows, desc="Processing videos"):
            result = process_single_video(
                row, data_root, output_dir, nframes,
                mask_area_ratio_range=mask_area_ratio_range,
            )
            if result:
                results.append(result)
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_single_video,
                    row, data_root, output_dir, nframes,
                    mask_area_ratio_range,
                ): row for row in rows
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
                result = future.result()
                if result:
                    results.append(result)
    
    # Save output CSV
    output_csv = os.path.join(output_dir, "training_data.csv")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_path', 'mask_path', 'ref_image_path', 'caption'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nDone! Generated {len(results)} samples")
    print(f"Output CSV: {output_csv}")
    print(f"Use with InpaintingDatasetFromCSV:")
    print(f"  --csv_file {output_csv}")
    print(f"  --data_root {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for video inpainting")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV with video paths")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for input videos")
    parser.add_argument("--nframes", type=int, default=81, help="Number of frames per video")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--mask_area_min", type=float, default=0.15, help="Min mask area ratio")
    parser.add_argument("--mask_area_max", type=float, default=0.40, help="Max mask area ratio")
    
    args = parser.parse_args()
    
    generate_training_data(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        data_root=args.data_root,
        nframes=args.nframes,
        num_workers=args.num_workers,
        mask_area_ratio_range=(args.mask_area_min, args.mask_area_max),
    )
