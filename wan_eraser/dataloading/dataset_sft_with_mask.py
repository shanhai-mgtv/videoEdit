"""
SFT Dataset with External Mask Input for Video Editing Training

This dataset combines:
1. Mask input from separate CSV (like dataset_s1.py) - reads mask from video files
2. Mask augmentation to generate ref_image and ref_mask (like dataset_inpaint.py)

Input:
    - Video CSV: video_path, prompt
    - Mask CSV: mask_path (video format, e.g., from SAM2)

Output:
    target_images: [F, C, H, W] - ground truth video frames
    masked_video: [F, C, H, W] - video with mask==1 region blacked out
    mask_video: [F, 1, H, W] - binary mask video (from external mask)
    mask_image: [1, H, W] - first frame mask
    ref_image: [C, H, W] - first frame foreground (augmented)
    ref_masked_image: [1, H, W] - augmented mask for reference
    caption: str - text prompt
    fps: float - video fps
"""

import random
import math
import csv
import os
import copy
from typing import Optional, Tuple, List
from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from decord import VideoReader
from torch.utils.data import Dataset
from torchvision import transforms
from moviepy import ImageSequenceClip

import sys
sys.path.append("/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/utils")
from saber_mask import random_affine_preserve_mask


# ============================================================================
# Aspect Ratio Configuration
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
    """Find the closest aspect ratio from predefined ratios."""
    aspect_ratio = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


# ============================================================================
# Debug Utilities
# ============================================================================

def save_video_frames(frames: List[np.ndarray], output_path: str, fps: int = 24):
    """Save list of numpy frames as video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", bitrate="10M", logger=None)


def save_debug_outputs(
    output_dir: str,
    sample_idx: int,
    video: np.ndarray,
    masked_video: np.ndarray,
    mask_video: np.ndarray,
    ref_image: np.ndarray,
    ref_image_aug: Optional[torch.Tensor] = None,
    ref_mask_aug: Optional[torch.Tensor] = None,
    caption: str = "",
    fps: float = 24,
):
    """Save generated training data for debugging/visualization."""
    sample_dir = Path(output_dir) / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original video
    save_video_frames([f for f in video], str(sample_dir / "original.mp4"), int(fps))
    
    # Save masked video
    save_video_frames([f for f in masked_video], str(sample_dir / "masked.mp4"), int(fps))
    
    # Save mask video
    mask_frames_rgb = []
    for i in range(mask_video.shape[0]):
        mask_frame = (mask_video[i] * 255).astype(np.uint8)
        if mask_frame.ndim == 2:
            mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2RGB)
        elif mask_frame.shape[-1] == 1:
            mask_frame = np.repeat(mask_frame, 3, axis=-1)
        mask_frames_rgb.append(mask_frame)
    save_video_frames(mask_frames_rgb, str(sample_dir / "mask.mp4"), int(fps))
    
    # Save first frame mask
    first_mask = (mask_video[0] * 255).astype(np.uint8)
    if first_mask.ndim == 3:
        first_mask = first_mask[:, :, 0]
    cv2.imwrite(str(sample_dir / "mask_first_frame.png"), first_mask)
    
    # Save reference image (foreground kept)
    cv2.imwrite(str(sample_dir / "ref_image.png"), cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))

    # Save augmented ref_image
    if ref_image_aug is not None:
        if torch.is_tensor(ref_image_aug):
            x = ref_image_aug.detach().cpu().float()
            x = x * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
            x = x.clamp(0, 1)
            x = (x * 255.0).round().byte()
            x = x.permute(1, 2, 0).contiguous().numpy()
            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)
            cv2.imwrite(str(sample_dir / "ref_image_aug.png"), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

    # Save augmented ref_mask
    if ref_mask_aug is not None:
        if torch.is_tensor(ref_mask_aug):
            m = ref_mask_aug.detach().cpu().float()
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            m = ((m > 0.5).byte().numpy() * 255)
            cv2.imwrite(str(sample_dir / "ref_mask_aug.png"), m)
    
    # Save caption
    with open(sample_dir / "caption.txt", "w") as f:
        f.write(caption)
    
    return {'sample_dir': str(sample_dir)}


# ============================================================================
# Main Dataset Class
# ============================================================================

class SFTDatasetWithMask(Dataset):
    """
    SFT Dataset that reads masks from external video files (like SAM2 output).
    
    Key features:
    1. Reads video from video_csv, mask from mask_csv (both in video format)
    2. Applies random_affine_preserve_mask for ref_image and ref_mask augmentation
    3. Outputs all necessary data for three-branch SFT training
    
    Args:
        args: config with csv_file_list, mask_csv_file_list, data_root, nframes, etc.
        save_debug: whether to save debug outputs
        debug_output_dir: directory for debug outputs
        debug_save_prob: probability of saving each sample
        augment_ref: whether to apply augmentation to ref_image and ref_mask
    """
    
    def __init__(
        self,
        args,
        save_debug: bool = False,
        debug_output_dir: str = "./debug_outputs",
        debug_save_prob: float = 0.01,
        augment_ref: bool = True,
    ):
        self.args = args
        self.repeat = 1
        self.nframes = args.nframes
        self.csv_file_list = args.csv_file_list
        self.mask_csv_file_list = args.mask_csv_file_list
        self.data_root = args.data_root

        # Debug/visualization settings
        self.save_debug = save_debug
        self.debug_output_dir = debug_output_dir
        self.debug_save_prob = debug_save_prob
        self.debug_counter = 0
        
        # Augmentation settings
        self.augment_ref = augment_ref
        self.aug_degrees = 10.0
        self.aug_scale_range = (0.8, 2.0)
        self.aug_hflip_prob = 0.5
        self.aug_shear_range = (-10.0, 10.0)
        
        self._load_metadata()
        self.mask_len = len(self.metadata_mask)
        print(f'[SFTDatasetWithMask] len of video metadata: {len(self.metadata)}')
        print(f'[SFTDatasetWithMask] len of mask metadata: {self.mask_len}')

    def _load_metadata(self):
        """Load video and mask CSV files."""
        self.metadata = []
        for csv_file in self.csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata.append(row)
        
        self.metadata_mask = []
        for csv_file in self.mask_csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata_mask.append(row)

    def __len__(self):
        return len(self.metadata) * self.repeat

    def _aug(self, frame, transform, state=None):
        """Apply transform with optional random state."""
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame) if transform is not None else frame

    def _mask_hw_resize(self, height_v, width_v, ori_ratio, mask_ratio):
        """Calculate mask resize parameters to match video dimensions."""
        if ori_ratio > mask_ratio:
            mask_resize_size = int(width_v * mask_ratio), width_v
            pad = "h"
            l_start = random.randint(0, max(0, height_v - mask_resize_size[0] - 1))
            pad_temp = mask_resize_size[0] + l_start
        elif ori_ratio < mask_ratio:
            mask_resize_size = height_v, int(height_v / mask_ratio)
            l_start = random.randint(0, max(0, width_v - mask_resize_size[1] - 1))
            pad_temp = mask_resize_size[1] + l_start
            pad = "w"
        else:
            mask_resize_size = height_v, width_v
            pad = "z"
            l_start = 0
            pad_temp = width_v
        mask_resize_size = [mask_resize_size[1], mask_resize_size[0]]  # for cv2 resize
        return [mask_resize_size, pad], [l_start, pad_temp]

    def _mask_padding_resize(self, mask, mask_res_d, hw, height_v, width_v):
        """Resize and pad mask to match video dimensions."""
        mask = cv2.resize(mask, mask_res_d[0])
        if mask_res_d[1] == "h":
            mask_temp = np.zeros((height_v, width_v, 3), dtype=mask.dtype)
            mask_temp[hw[0]:hw[1], :, :] = mask
        elif mask_res_d[1] == "w":
            mask_temp = np.zeros((height_v, width_v, 3), dtype=mask.dtype)
            mask_temp[:, hw[0]:hw[1], :] = mask
        else:
            mask_temp = mask
        return mask_temp

    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            
            # Read video metadata
            raw_data = self.metadata[index]
            vid_path = os.path.join(self.data_root, raw_data.get('video_path', raw_data.get('vid_path', '')))
            vid_caption = raw_data.get('prompt', raw_data.get('caption', ''))
            
            # Random select mask from mask pool
            index_mask = random.randint(0, self.mask_len - 1)
            raw_mask_data = self.metadata_mask[index_mask]
            mask_path = os.path.join(self.data_root, raw_mask_data.get('mask_path', ''))

            try:
                video_reader = VideoReader(vid_path)
                mask_reader = VideoReader(mask_path)
                
                # Handle short videos
                if len(video_reader) < self.nframes:
                    repeat_num_v = math.ceil(self.nframes / len(video_reader))
                    if random.random() >= 0.5:
                        temp_list = list(range(len(video_reader))) + list(range(len(video_reader)))[::-1][1:-1]
                        all_frames = temp_list * repeat_num_v
                    else:
                        all_frames = list(range(len(video_reader))) + [len(video_reader) - 1] * (self.nframes - len(video_reader) + 3)
                else:
                    all_frames = list(range(len(video_reader)))
                
                # Handle short masks
                if len(mask_reader) < self.nframes:
                    repeat_num_m = math.ceil(self.nframes / len(mask_reader))
                    if random.random() >= 0.5:
                        temp_list = list(range(len(mask_reader))) + list(range(len(mask_reader)))[::-1][1:-1]
                        mask_all_frames = temp_list * repeat_num_m
                    else:
                        mask_all_frames = list(range(len(mask_reader))) + [len(mask_reader) - 1] * (self.nframes - len(mask_reader) + 3)
                else:
                    mask_all_frames = list(range(len(mask_reader)))

                # Select random clip
                rand_idx = random.randint(0, max(0, len(all_frames) - self.nframes - 1))
                frame_indices = all_frames[rand_idx:rand_idx + self.nframes]
                
                mask_rand_idx = random.randint(0, max(0, len(mask_all_frames) - self.nframes - 1))
                mask_frame_indices = mask_all_frames[mask_rand_idx:mask_rand_idx + self.nframes]
                
                if len(frame_indices) < self.nframes or len(mask_frame_indices) < self.nframes:
                    print(f"vid frames: {len(frame_indices)}, mask frames: {len(mask_frame_indices)}")
                    index += 1
                    continue
                
                video = video_reader.get_batch(frame_indices).asnumpy()  # [F, H, W, C]
                masks = mask_reader.get_batch(mask_frame_indices).asnumpy()  # [F, H, W, C]
                fps = video_reader.get_avg_fps()
                break
                
            except Exception as e:
                print(f"Load data failed! video={vid_path}, mask={mask_path}, error={e}")
                index += 1
                continue

        assert video.shape[0] == self.nframes, f'{video.shape[0]}, self.nframes={self.nframes}'

        height_v, width_v, _ = video[0].shape
        height_m, width_m, _ = masks[0].shape
        ori_ratio = height_v / width_v
        mask_ratio = height_m / width_m
        
        # Calculate mask resize parameters
        mask_res_d, hw = self._mask_hw_resize(height_v, width_v, ori_ratio, mask_ratio)

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
        mask_video_frames = []
        masked_video_np = []
        mask_video_np = []
        
        state = torch.get_rng_state()
        
        for t in range(self.nframes):
            frame = video[t]
            
            # Process mask: resize and pad to match video
            mask_frame = masks[t]
            mask_frame = self._mask_padding_resize(mask_frame, mask_res_d, hw, height_v, width_v)
            mask_binary = (mask_frame > 5).astype(np.uint8)  # Binarize
            
            # Target image (ground truth)
            img = Image.fromarray(frame)
            target_img = self._aug(img, img_transform, state)
            target_images.append(target_img)
            
            # Masked video (mask==1 region blacked out)
            masked_frame = frame.copy()
            masked_frame = masked_frame * (1 - mask_binary)  # Black out mask==1 region
            masked_video_np.append(masked_frame)
            masked_img = Image.fromarray(masked_frame)
            masked_img = self._aug(masked_img, img_transform, state)
            masked_video_frames.append(masked_img)
            
            # Mask tensor
            mask_gray = mask_binary[:, :, 0] if mask_binary.ndim == 3 else mask_binary
            mask_video_np.append(mask_gray)
            mask_pil = Image.fromarray((mask_gray * 255).astype(np.uint8)).convert("L")
            mask_tensor = self._aug(mask_pil, mask_transform, state)
            mask_video_frames.append(mask_tensor)

        # Stack tensors
        target_images = torch.stack(target_images, dim=0)  # [F, C, H, W]
        masked_video = torch.stack(masked_video_frames, dim=0)  # [F, C, H, W]
        mask_video = torch.stack(mask_video_frames, dim=0)  # [F, 1, H, W]
        
        # Extract reference image (first frame with mask==1 region kept)
        first_frame = video[0]
        first_mask = mask_video_np[0]
        first_mask_3ch = np.stack([first_mask] * 3, axis=-1) if first_mask.ndim == 2 else first_mask
        first_frame_fg = first_frame.copy()
        first_frame_fg[first_mask_3ch < 0.5] = 0  # Keep only mask==1 region
        first_frame_fg_pil = Image.fromarray(first_frame_fg)
        ref_image = self._aug(first_frame_fg_pil, img_transform, state)  # [C, H, W]
        
        # Reference masked image is binary mask [1, H, W]
        ref_masked_image = mask_video[0].clone()  # [1, H, W]
        
        # Mask image (first frame mask)
        mask_image = mask_video[0]  # [1, H, W]

        # Apply augmentation to ref_image and ref_mask
        if self.augment_ref:
            ref_image_aug, ref_masked_image_aug, _ = random_affine_preserve_mask(
                ref_image,
                ref_masked_image,
                degrees=self.aug_degrees,
                scale_range=self.aug_scale_range,
                hflip_prob=self.aug_hflip_prob,
                shear_range=self.aug_shear_range,
            )
        else:
            ref_image_aug = ref_image
            ref_masked_image_aug = ref_masked_image

        # Debug saving
        if self.save_debug and random.random() < self.debug_save_prob:
            save_debug_outputs(
                output_dir=self.debug_output_dir,
                sample_idx=self.debug_counter,
                video=video,
                masked_video=np.stack(masked_video_np, axis=0),
                mask_video=np.stack(mask_video_np, axis=0),
                ref_image=first_frame_fg,
                ref_image_aug=ref_image_aug,
                ref_mask_aug=ref_masked_image_aug,
                caption=vid_caption,
                fps=fps,
            )
            self.debug_counter += 1
        
        outputs = {
            'target_images': target_images,          # [F, C, H, W]
            'masked_video': masked_video,            # [F, C, H, W]
            'mask_video': mask_video,                # [F, 1, H, W]
            'mask_image': mask_image,                # [1, H, W]
            'ref_image': ref_image_aug,              # [C, H, W] - augmented
            'ref_masked_image': ref_masked_image_aug, # [1, H, W] - binary mask, augmented
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs


# ============================================================================
# Demo Script
# ============================================================================

def demo():
    """
    Demo script to test the SFTDatasetWithMask.
    
    Usage:
        python dataset_sft_with_mask.py
    """
    from dataclasses import dataclass
    from typing import List
    import tempfile
    
    @dataclass
    class DemoArgs:
        csv_file_list: List[str]
        mask_csv_file_list: List[str]
        data_root: str
        nframes: int
    
    # Create dummy CSV files for testing
    print("=" * 60)
    print("SFTDatasetWithMask Demo")
    print("=" * 60)
    
    # Check if we have real data to test with
    test_video_csv = "/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/dataloading/test_video.csv"
    test_mask_csv = "/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/dataloading/test_mask.csv"
    
    # Print expected CSV format
    print("\n[Expected CSV Format]")
    print("\nVideo CSV (csv_file_list):")
    print("  video_path,prompt")
    print("  path/to/video1.mp4,\"A person walking\"")
    print("  path/to/video2.mp4,\"A car driving\"")
    
    print("\nMask CSV (mask_csv_file_list):")
    print("  mask_path")
    print("  path/to/mask1.mp4")
    print("  path/to/mask2.mp4")
    
    print("\n[Dataset Output Format]")
    print("  target_images: [F, C, H, W] - ground truth video")
    print("  masked_video: [F, C, H, W] - video with mask region blacked out")
    print("  mask_video: [F, 1, H, W] - binary mask video")
    print("  mask_image: [1, H, W] - first frame mask")
    print("  ref_image: [C, H, W] - first frame foreground (augmented)")
    print("  ref_masked_image: [1, H, W] - augmented mask for reference")
    print("  caption: str - text prompt")
    print("  fps: float - video fps")
    
    print("\n[Augmentation Applied to ref_image and ref_masked_image]")
    print("  - Random rotation: [-10°, 10°]")
    print("  - Random scale: [0.8, 2.0]")
    print("  - Random horizontal flip: 50%")
    print("  - Random shear: [-10°, 10°]")
    print("  - All transforms preserve mask region inside frame")
    
    # Try to run with real data if available
    try:
        # Look for existing CSV files
        possible_video_csvs = [
            "/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/dataloading/script/video_list.csv",
        ]
        possible_mask_csvs = [
            "/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/dataloading/script/mask_list.csv",
        ]
        
        video_csv = None
        mask_csv = None
        
        for p in possible_video_csvs:
            if os.path.exists(p):
                video_csv = p
                break
        
        for p in possible_mask_csvs:
            if os.path.exists(p):
                mask_csv = p
                break
        
        if video_csv and mask_csv:
            print(f"\n[Testing with real data]")
            print(f"  Video CSV: {video_csv}")
            print(f"  Mask CSV: {mask_csv}")
            
            args = DemoArgs(
                csv_file_list=[video_csv],
                mask_csv_file_list=[mask_csv],
                data_root="",
                nframes=17,
            )
            
            dataset = SFTDatasetWithMask(
                args,
                save_debug=True,
                debug_output_dir="./demo_output",
                debug_save_prob=1.0,
                augment_ref=True,
            )
            
            print(f"\n[Dataset Info]")
            print(f"  Total samples: {len(dataset)}")
            
            if len(dataset) > 0:
                print("\n[Loading first sample...]")
                sample = dataset[0]
                
                print("\n[Sample Shapes]")
                for key, value in sample.items():
                    if torch.is_tensor(value):
                        print(f"  {key}: {value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {type(value).__name__}")
                
                print("\n[Debug outputs saved to ./demo_output/]")
        else:
            print("\n[No test data found. Please provide valid CSV files.]")
            print("\nTo test the dataset, create CSV files with the format shown above.")
            
    except Exception as e:
        print(f"\n[Demo Error]: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    demo()
