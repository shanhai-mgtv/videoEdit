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
import sys
sys.path.append("/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/utils")
from saber_mask import generate_mask, random_affine_preserve_mask

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


# ============================================================================
# Utility Functions
# ============================================================================

ASPECT_RATIO_960 = {
    '0.25': [480., 1920.], '0.26': [480., 1856.], '0.27': [480., 1792.], '0.28': [480., 1728.],
    '0.32': [544., 1728.], '0.33': [544., 1664.], '0.35': [544., 1600.], '0.4':  [608., 1536.],
    '0.42':  [608., 1472.], '0.48': [480672., 1408.], '0.5': [672., 1344.], '0.52': [672., 1280.],
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
    ref_masked_image: np.ndarray = None,
    ref_image_aug: Optional[object] = None,
    ref_mask_aug: Optional[object] = None,
    caption: str = "",
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
    
    # Save reference image (foreground kept)
    cv2.imwrite(str(sample_dir / "ref_image.png"), cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
    
    # Save reference masked image (foreground removed)
    if ref_masked_image is not None:
        cv2.imwrite(str(sample_dir / "ref_masked_image.png"), cv2.cvtColor(ref_masked_image, cv2.COLOR_RGB2BGR))

    if ref_image_aug is not None:
        if torch.is_tensor(ref_image_aug):
            x = ref_image_aug.detach().cpu().float()
            x = x * 0.5 + 0.5
            x = x.clamp(0, 1)
            x = (x * 255.0).round().byte()
            x = x.permute(1, 2, 0).contiguous().numpy()
            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)
            cv2.imwrite(str(sample_dir / "ref_image_aug.png"), cv2.cvtColor(x, cv2.COLOR_RGB2BGR))

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
    
    # Return paths for CSV generation
    return {
        'sample_dir': str(sample_dir),
        'mask_path': str(sample_dir / "mask.png"),
        'ref_image_path': str(sample_dir / "ref_image.png"),
        'caption': caption,
    }

def augment_mask(
    mask: np.ndarray,
    rotation_range: Tuple[float, float] = (-10, 10),
    scale_range: Tuple[float, float] = (0.8, 2.0),
    flip_prob: float = 0.5,
    shear_range: Tuple[float, float] = (-10, 10),
) -> np.ndarray:
    """
    Apply random augmentation to mask.
    
    Args:
        mask: [H, W] binary mask (uint8)
        rotation_range: (min_deg, max_deg) rotation angle range
        scale_range: (min_scale, max_scale) scale range
        flip_prob: probability of horizontal flip
        shear_range: (min_deg, max_deg) shear angle range
    
    Returns:
        augmented mask: [H, W] uint8
    """
    h, w = mask.shape
    center = (w / 2, h / 2)
    
    # Random parameters
    angle = random.uniform(rotation_range[0], rotation_range[1])
    scale = random.uniform(scale_range[0], scale_range[1])
    shear_angle = random.uniform(shear_range[0], shear_range[1])
    do_flip = random.random() < flip_prob
    
    # Horizontal flip
    if do_flip:
        mask = np.fliplr(mask).copy()
    
    # Build affine transformation matrix
    # Rotation + Scale
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply shear
    shear_rad = np.deg2rad(shear_angle)
    shear_matrix = np.array([
        [1, np.tan(shear_rad), 0],
        [0, 1, 0]
    ], dtype=np.float32)
    
    # Combine: first shear, then rotation+scale
    M_rot_3x3 = np.vstack([M_rot, [0, 0, 1]])
    shear_3x3 = np.vstack([shear_matrix, [0, 0, 1]])
    combined = M_rot_3x3 @ shear_3x3
    M_combined = combined[:2, :]
    
    # Apply transformation
    augmented = cv2.warpAffine(
        mask, M_combined, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return augmented.astype(np.uint8)

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
        
        # Extract reference image (first frame with mask==1 kept, same transform as mask)
        first_frame = video[0]
        first_frame_fg = first_frame.copy()
        first_frame_fg[mask < 0.5] = 0  # Keep only mask==1 region
        first_frame_fg_pil = Image.fromarray(first_frame_fg)
        ref_image = self._aug(first_frame_fg_pil, img_transform, state)  # [C, H, W]
        
        # Reference masked image is binary mask [1, H, W] (same as mask_image)
        ref_masked_image = mask_tensor.clone()  # [1, H, W]
        
        # Mask image (first frame mask)
        mask_image = mask_video[0]  # [1, H, W]

        ref_image_aug, ref_masked_image_aug, _ = random_affine_preserve_mask(
            ref_image,
            ref_masked_image,
            degrees=10.0,
            scale_range=(0.8, 2.0),
            hflip_prob=0.5,
            shear_range=(-10.0, 10.0),
        )

        if self.save_debug and random.random() < self.debug_save_prob:
            save_debug_outputs(
                output_dir=self.debug_output_dir,
                sample_idx=self.debug_counter,
                video=video,
                masked_video=np.stack(masked_video_np, axis=0),
                mask=mask,
                ref_image=first_frame_fg,
                ref_masked_image=None,
                ref_image_aug=ref_image_aug,
                ref_mask_aug=ref_masked_image_aug,
                caption=vid_caption,
                fps=fps,
            )
            self.debug_counter += 1
        
        outputs = {
            'target_images': target_images,      # [F, C, H, W]
            'masked_video': masked_video,        # [F, C, H, W]
            'mask_video': mask_video,            # [F, 1, H, W]
            'mask_image': mask_image,            # [1, H, W]
            'ref_image': ref_image_aug,          # [C, H, W] - augmented
            'ref_masked_image': ref_masked_image_aug, # [1, H, W] - binary mask, augmented
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs


class PreGeneratedInpaintingDataset(Dataset):
    """
    Dataset for video inpainting training with pre-generated masks and reference images.
    
    Reads pre-generated data from generate_training_data.py output:
    - mask.png: static binary mask
    - ref_image.png: first frame foreground (mask region kept)
    
    Input CSV format (from generate_training_data.py):
        video_path: path to original video
        mask_path: relative path to mask.png
        ref_image_path: relative path to ref_image.png
        caption: text prompt

    Args:
        args: config with csv_file_list, data_root, nframes, etc.
        pregenerated_data_root: root directory for pre-generated data (mask/ref_image)
        save_debug: whether to save debug outputs
        debug_output_dir: directory for debug outputs
        debug_save_prob: probability of saving each sample
    """
    
    def __init__(
        self,
        args,
        pregenerated_data_root: str = None,
        save_debug: bool = False,
        debug_output_dir: str = "./debug_outputs",
        debug_save_prob: float = 0.01,
    ):
        self.args = args
        self.repeat = 1
        self.nframes = args.nframes
        self.csv_file_list = args.csv_file_list
        self.data_root = args.data_root
        self.pregenerated_data_root = pregenerated_data_root or args.data_root

        # Debug/visualization settings
        self.save_debug = save_debug
        self.debug_output_dir = debug_output_dir
        self.debug_save_prob = debug_save_prob
        self.debug_counter = 0
        
        self._load_metadata()
        print(f'[PreGeneratedInpaintingDataset] len of metadata: {len(self.metadata)}')

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
            
            # Read metadata
            raw_data = self.metadata[index]
            vid_path = os.path.join(self.data_root, raw_data.get('video_path', raw_data.get('vid_path', '')))
            mask_path = os.path.join(self.pregenerated_data_root, raw_data.get('mask_path', ''))
            ref_image_path = os.path.join(self.pregenerated_data_root, raw_data.get('ref_image_path', ''))
            vid_caption = raw_data.get('caption', raw_data.get('prompt', ''))

            try:
                # Load video
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
                
                # Load pre-generated mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    print(f"Failed to load mask: {mask_path}")
                    index += 1
                    continue
                mask = (mask > 127).astype(np.uint8)  # Binarize
                
                # Load pre-generated reference image
                ref_image_np = cv2.imread(ref_image_path)
                if ref_image_np is None:
                    print(f"Failed to load ref_image: {ref_image_path}")
                    index += 1
                    continue
                ref_image_np = cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB)
                
                break
                
            except Exception as e:
                print(f"Load data failed! video={vid_path}, error={e}")
                index += 1
                continue

        assert video.shape[0] == self.nframes, f'{video.shape[0]}, self.nframes={self.nframes}'

        height_v, width_v, _ = video[0].shape
        ori_ratio = height_v / width_v

        # Resize mask and ref_image to match video size if needed
        if mask.shape[0] != height_v or mask.shape[1] != width_v:
            mask = cv2.resize(mask, (width_v, height_v), interpolation=cv2.INTER_NEAREST)
        if ref_image_np.shape[0] != height_v or ref_image_np.shape[1] != width_v:
            ref_image_np = cv2.resize(ref_image_np, (width_v, height_v), interpolation=cv2.INTER_LINEAR)

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
        
        # Process pre-generated reference image
        ref_image_pil = Image.fromarray(ref_image_np)
        ref_image = self._aug(ref_image_pil, img_transform, state)  # [C, H, W]
        
        # Reference masked image is binary mask [1, H, W] (same as mask_image)
        ref_masked_image = mask_tensor.clone()  # [1, H, W]
        
        # Mask image (first frame mask)
        mask_image = mask_video[0]  # [1, H, W]

        # Apply mask augmentation using random_affine_preserve_mask
        ref_image_aug, ref_masked_image_aug, _ = random_affine_preserve_mask(
            ref_image,
            ref_masked_image,
            degrees=10.0,
            scale_range=(0.8, 2.0),
            hflip_prob=0.5,
            shear_range=(-10.0, 10.0),
        )

        if self.save_debug and random.random() < self.debug_save_prob:
            save_debug_outputs(
                output_dir=self.debug_output_dir,
                sample_idx=self.debug_counter,
                video=video,
                masked_video=np.stack(masked_video_np, axis=0),
                mask=mask,
                ref_image=ref_image_np,
                ref_masked_image=None,
                ref_image_aug=ref_image_aug,
                ref_mask_aug=ref_masked_image_aug,
                caption=vid_caption,
                fps=fps,
            )
            self.debug_counter += 1
        
        outputs = {
            'target_images': target_images,      # [F, C, H, W]
            'masked_video': masked_video,        # [F, C, H, W]
            'mask_video': mask_video,            # [F, 1, H, W]
            'mask_image': mask_image,            # [1, H, W]
            'ref_image': ref_image_aug,          # [C, H, W] - augmented
            'ref_masked_image': ref_masked_image_aug, # [1, H, W] - binary mask, augmented
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs


# ============================================================================
# Standalone Debug Script Entry Point
# ============================================================================
def _process_single_sample(args_tuple):
    """Worker function for multi-threaded processing."""
    idx, dataset, total = args_tuple
    try:
        sample = dataset[idx]
        return {
            'idx': idx,
            'success': True,
            'target_images': sample['target_images'].shape,
            'masked_video': sample['masked_video'].shape,
            'mask_video': sample['mask_video'].shape,
            'mask_image': sample['mask_image'].shape,
            'ref_image': sample['ref_image'].shape,
            'ref_masked_image': sample['ref_masked_image'].shape,
            'caption': sample['caption'][:50] if sample['caption'] else '',
        }
    except Exception as e:
        return {'idx': idx, 'success': False, 'error': str(e)}


def debug_dataset(
    csv_file: str,
    data_root: str,
    output_dir: str = "./debug_outputs",
    num_samples: int = 5,
    nframes: int = 81,
    num_workers: int = 1,
    use_pregenerated: bool = False,
    pregenerated_data_root: str = None,
):
    from dataclasses import dataclass
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    import glob
    
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
    
    if use_pregenerated:
        print(f"Using PreGeneratedInpaintingDataset with root: {pregenerated_data_root}")
        dataset = PreGeneratedInpaintingDataset(
            args,
            pregenerated_data_root=pregenerated_data_root,
            save_debug=True,
            debug_output_dir=output_dir,
            debug_save_prob=1.0,
        )
    else:
        print("Using InpaintingDataset with online mask generation")
        dataset = InpaintingDataset(
            args,
            save_debug=True,
            debug_output_dir=output_dir,
            debug_save_prob=1.0,
        )
    
    actual_samples = min(num_samples, len(dataset))
    print(f"Dataset length: {len(dataset)}")
    print(f"Processing {actual_samples} samples with {num_workers} workers")
    print(f"Output directory: {output_dir}")
    
    # Read original CSV to get video paths
    original_metadata = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_metadata.append(row)
    
    if num_workers <= 1:
        # Sequential processing
        for i in tqdm(range(actual_samples), desc="Processing samples"):
            sample = dataset[i]
    else:
        # Multi-threaded processing
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_sample, (i, dataset, actual_samples)): i 
                for i in range(actual_samples)
            }
            
            for future in tqdm(as_completed(futures), total=actual_samples, desc="Processing samples"):
                result = future.result()
                results.append(result)
                if not result['success']:
                    print(f"  [Sample {result['idx']+1}] FAILED: {result['error']}")
        
        success_count = sum(1 for r in results if r['success'])
        print(f"\nProcessed {success_count}/{actual_samples} samples successfully")
    
    # Generate CSV from saved outputs
    print("\nGenerating training_data.csv...")
    output_path = Path(output_dir)
    csv_rows = []
    
    # Find all sample directories
    sample_dirs = sorted(glob.glob(str(output_path / "sample_*")))
    print(f"Found {len(sample_dirs)} sample directories")
    
    for sample_dir in sample_dirs:
        sample_dir = Path(sample_dir)
        mask_path = sample_dir / "mask.png"
        ref_image_path = sample_dir / "ref_image.png"
        caption_path = sample_dir / "caption.txt"
        
        if not mask_path.exists() or not ref_image_path.exists():
            continue
        
        # Read caption
        caption = ""
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
        
        # Get sample index from directory name
        sample_idx = int(sample_dir.name.split('_')[1])
        
        # Get original video path if available
        video_path = ""
        if sample_idx < len(original_metadata):
            video_path = original_metadata[sample_idx].get('video_path', 
                         original_metadata[sample_idx].get('vid_path', ''))
        
        # Use relative paths
        rel_mask = os.path.relpath(mask_path, output_dir)
        rel_ref = os.path.relpath(ref_image_path, output_dir)
        
        csv_rows.append({
            'video_path': video_path,
            'mask_path': rel_mask,
            'ref_image_path': rel_ref,
            'caption': caption,
        })
    
    # Write CSV
    csv_output_path = output_path / "training_data.csv"
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_path', 'mask_path', 'ref_image_path', 'caption'])
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\nGenerated CSV with {len(csv_rows)} samples: {csv_output_path}")
    print(f"\nDone! Check outputs in {output_dir}")
    print(f"\nTo use with PreGeneratedInpaintingDataset:")
    print(f"  csv_file_list: ['{csv_output_path}']")
    print(f"  pregenerated_data_root: '{output_dir}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug InpaintingDataset")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for videos")
    parser.add_argument("--output_dir", type=str, default="./debug_outputs", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to save")
    parser.add_argument("--nframes", type=int, default=81, help="Number of frames")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--use_pregenerated", action="store_true", help="Use pre-generated mask/ref_image")
    parser.add_argument("--pregenerated_data_root", type=str, default=None, help="Root directory for pre-generated data")
    
    args = parser.parse_args()
    
    debug_dataset(
        csv_file=args.csv_file,
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        nframes=args.nframes,
        num_workers=args.num_workers,
        use_pregenerated=args.use_pregenerated,
        pregenerated_data_root=args.pregenerated_data_root,
    )
