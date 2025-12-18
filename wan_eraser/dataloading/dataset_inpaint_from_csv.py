"""
Inpainting Dataset for Video Editing Training - CSV Version

Input: Pre-generated data from CSV (video + mask_video + ref_image + caption)
Output: Reads all training inputs from disk

CSV Format:
    video_path: path to original video
    mask_video_path: path to mask video (binary mask per frame)
    ref_image_path: path to reference image (foreground region)
    caption: text prompt

This version reads pre-generated masks from disk instead of generating them on the fly.
Use generate_training_data.py to pre-generate the data.
"""

import random
import math
import csv
import os
from typing import Optional, Tuple, List
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from decord import VideoReader
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.append("/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/utils")
from saber_mask import random_affine_preserve_mask

# ============================================================================
# Utility Functions
# ============================================================================

ASPECT_RATIO_960 = {
    '0.25': [480., 1920.], '0.26': [480., 1856.], '0.27': [480., 1792.], '0.28': [480., 1728.],
    '0.32': [544., 1728.], '0.33': [544., 1664.], '0.35': [544., 1600.], '0.4':  [608., 1536.],
    '0.42':  [608., 1472.], '0.48': [672., 1408.], '0.5': [672., 1344.], '0.52': [672., 1280.],
    '0.57': [736., 1280.], '0.6': [736., 1216.], '0.68': [800., 1152.], '0.72': [800., 1088.],
    '0.78': [864., 1088.], '0.82': [864.,  960.], '0.88': [928.,  960.], '0.94': [928.,  896.],
    '1.0':  [960.,  960.], '1.07': [960.,  896.], '1.13': [1024.,  896.], '1.21': [1024.,  832.],
    '1.29': [1088.,  832.], '1.38': [1088.,  768.], '1.46': [1152.,  768.], '1.67': [1216.,  704.],
    '1.75': [1280.,  704.], '2.0': [1344.,  640.], '2.09': [1408.,  640.], '2.4': [1472.,  576.],
    '2.5': [1536.,  576.], '2.89': [1600.,  544.], '3.0': [1664.,  544.], '3.11': [1728.,  544.],
    '3.62': [1792.,  512.], '3.75': [1856.,  512.], '3.88': [1920.,  512.], '4.0': [1920.,  480.],
}


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


# ============================================================================
# Dataset Class - Read from Pre-generated CSV
# ============================================================================

class InpaintingDatasetFromCSV(Dataset):
    """
    Dataset for video inpainting training - reads pre-generated data from CSV.
    
    Input: CSV with paths to video, mask_video, ref_image, caption
    Output: Training tensors ready for model

    CSV columns required:
        - video_path: path to original video (.mp4)
        - mask_video_path: path to mask video (.mp4, grayscale)
        - ref_image_path: path to reference image (.png, RGB)
        - caption: text prompt

    Args:
        args: config with csv_file_list, data_root, nframes, etc.
    """
    
    def __init__(
        self,
        args,
        augment_ref: bool = True,  # Whether to augment ref_image
    ):
        self.args = args
        self.repeat = 1
        self.nframes = args.nframes
        self.csv_file_list = args.csv_file_list
        self.data_root = args.data_root
        self.augment_ref = augment_ref
        
        self._load_metadata()
        print(f'[InpaintingDatasetFromCSV] len of metadata: {len(self.metadata)}')

    def _load_metadata(self):
        self.metadata = []
        for csv_file in self.csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Validate required fields (support both mask_path and mask_video_path)
                    has_mask = 'mask_path' in row or 'mask_video_path' in row
                    if 'video_path' in row and has_mask and 'ref_image_path' in row:
                        self.metadata.append(row)
                    else:
                        print(f"Warning: skipping row missing required fields: {row}")

    def __len__(self):
        return len(self.metadata) * self.repeat

    def _aug(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame) if transform is not None else frame

    def _load_video(self, video_path: str, nframes: int) -> Tuple[np.ndarray, float]:
        """Load video and return frames + fps."""
        video_reader = VideoReader(video_path)
        
        # Handle short videos by repeating frames
        if len(video_reader) < nframes:
            repeat_num = math.ceil(nframes / len(video_reader))
            if random.random() >= 0.5:
                temp_list = list(range(len(video_reader))) + list(range(len(video_reader)))[::-1][1:-1]
                all_frames = temp_list * repeat_num
            else:
                all_frames = list(range(len(video_reader))) + [len(video_reader) - 1] * (nframes - len(video_reader) + 3)
        else:
            all_frames = list(range(len(video_reader)))

        # Select random clip
        rand_idx = random.randint(0, max(0, len(all_frames) - nframes - 1))
        frame_indices = all_frames[rand_idx:rand_idx + nframes]
        
        video = video_reader.get_batch(frame_indices).asnumpy()  # [F, H, W, C]
        fps = video_reader.get_avg_fps()
        
        return video, fps, rand_idx

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load static mask image (grayscale)."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        # Binarize
        mask = (mask > 127).astype(np.uint8)
        return mask

    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            
            # Read metadata
            raw_data = self.metadata[index]
            vid_path = os.path.join(self.data_root, raw_data['video_path'])
            # Support both mask_path (static) and mask_video_path (legacy)
            mask_path_key = 'mask_path' if 'mask_path' in raw_data else 'mask_video_path'
            mask_path = os.path.join(self.data_root, raw_data[mask_path_key])
            ref_img_path = os.path.join(self.data_root, raw_data['ref_image_path'])
            vid_caption = raw_data.get('caption', raw_data.get('prompt', ''))

            try:
                # Load original video
                video, fps, start_idx = self._load_video(vid_path, self.nframes)
                
                # Load static mask image
                mask = self._load_mask(mask_path)
                
                # Load reference image
                ref_img_raw = cv2.imread(ref_img_path)
                if ref_img_raw is None:
                    raise ValueError(f"Failed to load ref image: {ref_img_path}")
                ref_img_raw = cv2.cvtColor(ref_img_raw, cv2.COLOR_BGR2RGB)
                
                break
                
            except Exception as e:
                print(f"Load failed! video={vid_path}, error={e}")
                index += 1
                continue

        assert video.shape[0] == self.nframes, f'{video.shape[0]}, self.nframes={self.nframes}'

        height_v, width_v, _ = video[0].shape
        ori_ratio = height_v / width_v

        # Compute target size
        closest_size, closest_ratio = get_closest_ratio(height_v, width_v, ASPECT_RATIO_960)
        closest_size = list(map(lambda x: int(x), closest_size))
        
        if closest_ratio > ori_ratio:
            resize_size = height_v, int(height_v / closest_ratio)
        else:
            resize_size = int(width_v * closest_ratio), width_v

        # Define transforms
        img_transform = transforms.Compose([
            transforms.RandomCrop(resize_size),
            transforms.Resize(closest_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        mask_transform = transforms.Compose([
            transforms.RandomCrop(resize_size),
            transforms.Resize(closest_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Process frames with same random crop for all
        target_images = []
        masked_video_frames = []
        
        state = torch.get_rng_state()
        
        for t in range(self.nframes):
            frame = video[t]
            
            # Target image (ground truth)
            img = Image.fromarray(frame)
            target_img = self._aug(img, img_transform, state)
            target_images.append(target_img)
            
            # Masked video (mask==1 region blacked out, using static mask)
            masked_frame = frame.copy()
            masked_frame[mask > 0.5] = 0
            masked_img = Image.fromarray(masked_frame)
            masked_img = self._aug(masked_img, img_transform, state)
            masked_video_frames.append(masked_img)
        
        # Process static mask (same for all frames)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        mask_tensor = self._aug(mask_pil, mask_transform, state)
        
        # Stack tensors
        target_images = torch.stack(target_images, dim=0)  # [F, C, H, W]
        masked_video = torch.stack(masked_video_frames, dim=0)  # [F, C, H, W]
        mask_video = mask_tensor.unsqueeze(0).repeat(self.nframes, 1, 1, 1)  # [F, 1, H, W]
        
        # Process reference image
        ref_img_pil = Image.fromarray(ref_img_raw)
        # Resize ref image to match video size first
        ref_img_resized = ref_img_pil.resize((width_v, height_v), Image.BICUBIC)
        ref_image = self._aug(ref_img_resized, img_transform, state)  # [C, H, W]
        
        # Reference masked image is static mask
        ref_masked_image = mask_tensor.clone()  # [1, H, W]
        
        # Mask image (first frame mask)
        mask_image = mask_video[0]  # [1, H, W]

        # Apply mask augmentation if enabled
        if self.augment_ref:
            ref_image_aug, ref_masked_image_aug, _ = random_affine_preserve_mask(
                ref_image,
                ref_masked_image,
                degrees=10.0,
                scale_range=(0.8, 2.0),
                hflip_prob=0.5,
                shear_range=(-10.0, 10.0),
            )
        else:
            ref_image_aug = ref_image
            ref_masked_image_aug = ref_masked_image
        
        outputs = {
            'target_images': target_images,      # [F, C, H, W]
            'masked_video': masked_video,        # [F, C, H, W]
            'mask_video': mask_video,            # [F, 1, H, W]
            'mask_image': mask_image,            # [1, H, W]
            'ref_image': ref_image_aug,          # [C, H, W]
            'ref_masked_image': ref_masked_image_aug,  # [1, H, W]
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs


# ============================================================================
# Debug Script
# ============================================================================

def debug_dataset(
    csv_file: str,
    data_root: str,
    output_dir: str = "./debug_outputs",
    num_samples: int = 5,
    nframes: int = 81,
):
    """Debug the dataset by loading and printing sample info."""
    from dataclasses import dataclass
    
    @dataclass
    class DebugArgs:
        csv_file_list: List[str]
        data_root: str
        nframes: int
    
    args = DebugArgs(
        csv_file_list=[csv_file],
        data_root=data_root,
        nframes=nframes,
    )
    
    dataset = InpaintingDatasetFromCSV(args)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Loading {num_samples} samples...")
    
    for i in range(min(num_samples, len(dataset))):
        print(f"Processing sample {i+1}/{num_samples}...")
        sample = dataset[i]
        print(f"  target_images: {sample['target_images'].shape}")
        print(f"  masked_video: {sample['masked_video'].shape}")
        print(f"  mask_video: {sample['mask_video'].shape}")
        print(f"  mask_image: {sample['mask_image'].shape}")
        print(f"  ref_image: {sample['ref_image'].shape}")
        print(f"  ref_masked_image: {sample['ref_masked_image'].shape}")
        print(f"  caption: {sample['caption'][:50]}...")
    
    print(f"\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug InpaintingDatasetFromCSV")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file with pre-generated data")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--output_dir", type=str, default="./debug_outputs", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to load")
    parser.add_argument("--nframes", type=int, default=81, help="Number of frames")
    
    args = parser.parse_args()
    
    debug_dataset(
        csv_file=args.csv_file,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        nframes=args.nframes,
    )
