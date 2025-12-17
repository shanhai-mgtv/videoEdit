"""
Inpainting Dataset for Video Editing Training

Input: Only video + caption
Output: Automatically generates all training inputs using mask generation strategy

Outputs:
    target_images: [F, C, H, W] - ground truth video frames
    masked_video: [F, C, H, W] - video with mask==1 region blacked out
    mask_video: [F, 1, H, W] - binary mask video (same mask for all frames)
    mask_image: [1, H, W] - first frame mask
    ref_image: [C, H, W] - first frame foreground (cropped by bbox)
    caption: str - text prompt
    fps: float - video fps

The mask is automatically generated using generate_mask() from mask_generator_and_augmentation.py
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
from moviepy import ImageSequenceClip


# ============================================================================
# Mask Generation Functions (from mask_generator_and_augmentation.py)
# ============================================================================

def neighbors8(y: int, x: int, h: int, w: int):
    """Return 8-connected neighbors within bounds."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                yield ny, nx


def neighbor_count(mask: np.ndarray, y: int, x: int, value: int = 1) -> int:
    """Count 8-connected neighbors with given value."""
    h, w = mask.shape
    cnt = 0
    for ny, nx in neighbors8(y, x, h, w):
        if mask[ny, nx] == value:
            cnt += 1
    return cnt


def simple_point_candidates(mask: np.ndarray, foreground: bool = True):
    """
    Find pixels that can be toggled without breaking connectivity.
    foreground=True: candidates to remove (shrink)
    foreground=False: candidates to add (grow)
    """
    h, w = mask.shape
    target_val = 1 if foreground else 0
    candidates = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] != target_val:
                continue
            n_same = neighbor_count(mask, y, x, target_val)
            n_opp = 8 - n_same if (y > 0 and y < h - 1 and x > 0 and x < w - 1) else (
                len(list(neighbors8(y, x, h, w))) - n_same
            )
            if foreground:
                if n_same >= 1 and n_opp >= 1:
                    candidates.append((y, x))
            else:
                if n_same >= 1:
                    candidates.append((y, x))
    return candidates


def grow_to(mask: np.ndarray, target_area: int) -> np.ndarray:
    """Grow mask until it reaches target_area."""
    mask = mask.copy()
    while mask.sum() < target_area:
        cands = simple_point_candidates(mask, foreground=False)
        if not cands:
            break
        y, x = random.choice(cands)
        mask[y, x] = 1
    return mask


def shrink_to(mask: np.ndarray, target_area: int) -> np.ndarray:
    """Shrink mask until it reaches target_area."""
    mask = mask.copy()
    while mask.sum() > target_area:
        cands = simple_point_candidates(mask, foreground=True)
        if not cands:
            break
        y, x = random.choice(cands)
        mask[y, x] = 0
    return mask


def generate_mask(
    h: int,
    w: int,
    area_ratio_range: Tuple[float, float] = (0.1, 0.5),
    shape_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a binary mask with area in given ratio range.
    
    Args:
        h, w: mask dimensions
        area_ratio_range: (min_ratio, max_ratio) for mask area
        shape_types: list of shape types to choose from
        seed: random seed
        
    Returns:
        mask: [H, W] uint8 array with values in {0, 1}
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if shape_types is None:
        shape_types = ["ellipse", "superellipse", "concave_polygon", "centered_rectangle"]
    
    shape_type = random.choice(shape_types)
    total_pixels = h * w
    A_lo = int(area_ratio_range[0] * total_pixels)
    A_hi = int(area_ratio_range[1] * total_pixels)
    target_area = random.randint(A_lo, A_hi)
    
    # Generate center position
    cy = random.randint(h // 4, 3 * h // 4)
    cx = random.randint(w // 4, 3 * w // 4)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if shape_type == "ellipse":
        # Random ellipse
        target_ratio = target_area / total_pixels
        base_radius = np.sqrt(target_ratio * h * w / np.pi)
        ry = int(base_radius * random.uniform(0.7, 1.3))
        rx = int(base_radius * random.uniform(0.7, 1.3))
        ry = max(10, min(ry, h // 2 - 10))
        rx = max(10, min(rx, w // 2 - 10))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
        
    elif shape_type == "superellipse":
        # Superellipse (rounded rectangle)
        target_ratio = target_area / total_pixels
        base_size = np.sqrt(target_ratio * h * w)
        ry = int(base_size * random.uniform(0.5, 1.0))
        rx = int(base_size * random.uniform(0.5, 1.0))
        ry = max(20, min(ry, h // 2 - 10))
        rx = max(20, min(rx, w // 2 - 10))
        n = random.uniform(2.5, 4.0)
        
        Y, X = np.ogrid[:h, :w]
        Y = (Y - cy) / ry
        X = (X - cx) / rx
        inside = (np.abs(X) ** n + np.abs(Y) ** n) <= 1
        mask[inside] = 1
        
    elif shape_type == "concave_polygon":
        # Random polygon
        num_vertices = random.randint(5, 8)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, num_vertices))
        target_ratio = target_area / total_pixels
        base_radius = np.sqrt(target_ratio * h * w / np.pi) * 1.2
        
        radii = base_radius * np.random.uniform(0.6, 1.0, num_vertices)
        vertices = []
        for angle, radius in zip(angles, radii):
            vx = int(cx + radius * np.cos(angle))
            vy = int(cy + radius * np.sin(angle))
            vx = max(5, min(w - 5, vx))
            vy = max(5, min(h - 5, vy))
            vertices.append([vx, vy])
        vertices = np.array(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 1)
        
    elif shape_type == "centered_rectangle":
        # Centered rectangle
        target_ratio = target_area / total_pixels
        base_size = np.sqrt(target_ratio * h * w)
        rh = int(base_size * random.uniform(0.7, 1.3))
        rw = int(base_size * random.uniform(0.7, 1.3))
        rh = max(20, min(rh, h - 20))
        rw = max(20, min(rw, w - 20))
        
        y1 = max(0, cy - rh // 2)
        y2 = min(h, cy + rh // 2)
        x1 = max(0, cx - rw // 2)
        x2 = min(w, cx + rw // 2)
        mask[y1:y2, x1:x2] = 1
    
    # Adjust area if needed
    current_area = mask.sum()
    if current_area < A_lo:
        mask = grow_to(mask, A_lo)
    elif current_area > A_hi:
        mask = shrink_to(mask, A_hi)
    
    return mask


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


def get_mask_bbox(mask: np.ndarray, padding: int = 10) -> Tuple[int, int, int, int]:
    """Get bounding box of mask==1 region with optional padding."""
    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return (0, mask.shape[0] - 1, 0, mask.shape[1] - 1)
    
    h, w = mask.shape
    y_min = max(0, int(ys.min()) - padding)
    y_max = min(h - 1, int(ys.max()) + padding)
    x_min = max(0, int(xs.min()) - padding)
    x_max = min(w - 1, int(xs.max()) + padding)
    
    return (y_min, y_max, x_min, x_max)


def crop_foreground_by_bbox(
    frame: np.ndarray,
    mask: np.ndarray,
    padding: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    """Crop the foreground region (mask==1) from frame using bounding box."""
    y_min, y_max, x_min, x_max = get_mask_bbox(mask, padding)
    
    cropped_frame = frame[y_min:y_max+1, x_min:x_max+1].copy()
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1].copy()
    
    # Black out background in cropped region
    cropped_frame[cropped_mask < 0.5] = 0
    
    return cropped_frame, cropped_mask, (y_min, y_max, x_min, x_max)


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
    mask: np.ndarray,
    ref_image: np.ndarray,
    caption: str,
    fps: float = 24,
):
    """
    Save generated training data for debugging/visualization.
    
    Args:
        output_dir: directory to save outputs
        sample_idx: sample index
        video: [F, H, W, C] original video
        masked_video: [F, H, W, C] masked video
        mask: [H, W] binary mask
        ref_image: [H, W, C] reference image
        caption: text caption
        fps: video fps
    """
    sample_dir = Path(output_dir) / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original video
    save_video_frames([f for f in video], str(sample_dir / "original.mp4"), int(fps))
    
    # Save masked video
    save_video_frames([f for f in masked_video], str(sample_dir / "masked.mp4"), int(fps))
    
    # Save mask video (same mask repeated)
    mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    save_video_frames([mask_rgb for _ in range(len(video))], str(sample_dir / "mask.mp4"), int(fps))
    
    # Save mask image
    cv2.imwrite(str(sample_dir / "mask.png"), mask * 255)
    
    # Save reference image
    cv2.imwrite(str(sample_dir / "ref_image.png"), cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
    
    # Save caption
    with open(sample_dir / "caption.txt", "w") as f:
        f.write(caption)
    
    print(f"Saved debug outputs to {sample_dir}")


# ============================================================================
# Dataset Class
# ============================================================================

class InpaintingDataset(Dataset):
    """
    Dataset for video inpainting training.
    
    Input: Only video + caption (from CSV)
    Output: Automatically generates mask, masked_video, ref_image

    Args:
        args: config with csv_file_list, data_root, nframes, etc.
        save_debug: whether to save debug outputs
        debug_output_dir: directory for debug outputs
        debug_save_prob: probability of saving each sample (to avoid saving all)
    """
    
    def __init__(
        self,
        args,
        save_debug: bool = False,
        debug_output_dir: str = "./debug_outputs",
        debug_save_prob: float = 0.01,
    ):
        self.args = args
        self.repeat = 1
        self.nframes = args.nframes
        self.csv_file_list = args.csv_file_list
        self.data_root = args.data_root

        # Debug/visualization settings
        self.save_debug = save_debug
        self.debug_output_dir = debug_output_dir
        self.debug_save_prob = debug_save_prob
        self.debug_counter = 0
        
        # Mask generation settings
        self.mask_area_ratio_range = (0.15, 0.40)
        self.mask_shape_types = ["ellipse", "superellipse", "concave_polygon", "centered_rectangle"]
        
        self._load_metadata()
        print(f'len of metadata: {len(self.metadata)}')

    def _load_metadata(self):
        self.metadata = []
        for csv_file in self.csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata.append(row)

    def __len__(self):
        return len(self.metadata) * self.repeat

    def _aug(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame) if transform is not None else frame

    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            
            # Read video metadata (only video + caption)
            raw_data = self.metadata[index]
            vid_path = os.path.join(self.data_root, raw_data.get('video_path', raw_data.get('vid_path', '')))
            vid_caption = raw_data.get('prompt', raw_data.get('caption', ''))

            try:
                video_reader = VideoReader(vid_path)
                
                # Handle short videos by repeating frames
                if len(video_reader) < self.nframes:
                    repeat_num = math.ceil(self.nframes / len(video_reader))
                    if random.random() >= 0.5:
                        temp_list = list(range(len(video_reader))) + list(range(len(video_reader)))[::-1][1:-1]
                        all_frames = temp_list * repeat_num
                    else:
                        all_frames = list(range(len(video_reader))) + [len(video_reader) - 1] * (self.nframes - len(video_reader) + 3)
                else:
                    all_frames = list(range(len(video_reader)))

                # Select random clip
                rand_idx = random.randint(0, max(0, len(all_frames) - self.nframes - 1))
                frame_indices = all_frames[rand_idx:rand_idx + self.nframes]
                
                if len(frame_indices) < self.nframes:
                    print(f"vid frames are {len(frame_indices)}")
                    index += 1
                    continue
                
                video = video_reader.get_batch(frame_indices).asnumpy()  # [F, H, W, C]
                fps = video_reader.get_avg_fps()
                break
                
            except Exception as e:
                print(f"Load video failed! path={vid_path}, error={e}")
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

        # ============================================
        # Generate mask automatically
        # ============================================
        mask = generate_mask(
            h=height_v,
            w=width_v,
            area_ratio_range=self.mask_area_ratio_range,
            shape_types=self.mask_shape_types,
        )

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
        masked_video_np = []  # For debug saving
        
        state = torch.get_rng_state()
        
        for t in range(self.nframes):
            frame = video[t]
            
            # Target image (ground truth)
            img = Image.fromarray(frame)
            target_img = self._aug(img, img_transform, state)
            target_images.append(target_img)
            
            # Masked video (mask==1 region blacked out)
            masked_frame = frame.copy()
            masked_frame[mask > 0.5] = 0  # Black out mask==1 region
            masked_video_np.append(masked_frame)
            masked_img = Image.fromarray(masked_frame)
            masked_img = self._aug(masked_img, img_transform, state)
            masked_video_frames.append(masked_img)

        # Process mask (same for all frames)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
        mask_tensor = self._aug(mask_pil, mask_transform, state)
        
        # Stack tensors
        target_images = torch.stack(target_images, dim=0)  # [F, C, H, W]
        masked_video = torch.stack(masked_video_frames, dim=0)  # [F, C, H, W]
        mask_video = mask_tensor.unsqueeze(0).repeat(self.nframes, 1, 1, 1)  # [F, 1, H, W]
        
        # Extract reference image (first frame foreground cropped by bbox)
        first_frame = video[0]
        ref_crop, ref_mask_crop, bbox = crop_foreground_by_bbox(first_frame, mask, padding=10)
        
        # Resize ref image to match target size
        ref_h, ref_w = closest_size
        ref_crop_pil = Image.fromarray(ref_crop)
        ref_transform = transforms.Compose([
            transforms.Resize((ref_h, ref_w), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        ref_image = ref_transform(ref_crop_pil)  # [C, H, W]
        
        # Mask image (first frame mask)
        mask_image = mask_video[0]  # [1, H, W]

        # ============================================
        # Save debug outputs if enabled
        # ============================================
        if self.save_debug and random.random() < self.debug_save_prob:
            save_debug_outputs(
                output_dir=self.debug_output_dir,
                sample_idx=self.debug_counter,
                video=video,
                masked_video=np.stack(masked_video_np, axis=0),
                mask=mask,
                ref_image=ref_crop,
                caption=vid_caption,
                fps=fps,
            )
            self.debug_counter += 1

        outputs = {
            'target_images': target_images,      # [F, C, H, W]
            'masked_video': masked_video,        # [F, C, H, W]
            'mask_video': mask_video,            # [F, 1, H, W]
            'mask_image': mask_image,            # [1, H, W]
            'ref_image': ref_image,              # [C, H, W]
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs


# ============================================================================
# Standalone Debug Script Entry Point
# ============================================================================

def debug_dataset(
    csv_file: str,
    data_root: str,
    output_dir: str = "./debug_outputs",
    num_samples: int = 5,
    nframes: int = 81,
):
    """
    Standalone function to debug/visualize dataset outputs.
    
    Usage:
        python dataset_inpaint.py --csv_file /path/to/data.csv --data_root /path/to/videos --output_dir ./debug --num_samples 5
    """
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
    
    dataset = InpaintingDataset(
        args,
        save_debug=True,
        debug_output_dir=output_dir,
        debug_save_prob=1.0,  # Save all samples in debug mode
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Saving {num_samples} samples to {output_dir}")
    
    for i in range(min(num_samples, len(dataset))):
        print(f"Processing sample {i+1}/{num_samples}...")
        sample = dataset[i]
        print(f"  target_images: {sample['target_images'].shape}")
        print(f"  masked_video: {sample['masked_video'].shape}")
        print(f"  mask_video: {sample['mask_video'].shape}")
        print(f"  mask_image: {sample['mask_image'].shape}")
        print(f"  ref_image: {sample['ref_image'].shape}")
        print(f"  caption: {sample['caption'][:50]}...")
    
    print(f"\nDone! Check outputs in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug InpaintingDataset")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for videos")
    parser.add_argument("--output_dir", type=str, default="./debug_outputs", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to save")
    parser.add_argument("--nframes", type=int, default=81, help="Number of frames")
    
    args = parser.parse_args()
    
    debug_dataset(
        csv_file=args.csv_file,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        nframes=args.nframes,
    )
