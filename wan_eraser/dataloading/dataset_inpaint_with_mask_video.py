"""
Inpainting Dataset with Mask Video Path Support

Input: video + mask_video + caption (from CSV)
Output: Training inputs with ref from any overlapping frame

Outputs:
    target_images: [F, C, H, W] - ground truth video frames
    masked_video: [F, C, H, W] - video with mask==1 region blacked out
    mask_video: [F, 1, H, W] - binary mask video (loaded from mask_video path)
    mask_image: [1, H, W] - first frame mask
    ref_image: [C, H, W] - random frame foreground (mask region kept), augmented
    ref_masked_image: [1, H, W] - reference mask, augmented
    caption: str - text prompt
    fps: float - video fps

Key differences from dataset_inpaint.py:
    1. Mask video is loaded from path instead of generated
    2. Reference frame is randomly selected from any frame (not just first)
    3. ref_image and ref_masked_image are augmented together
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

_iclight_relight_module = None
def get_iclight_relight():
    global _iclight_relight_module
    if _iclight_relight_module is None:
        sys.path.append("/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/utils/IC-Light")
        from iclight_relight import get_iclight_instance
        _iclight_relight_module = get_iclight_instance()
    return _iclight_relight_module


# ============================================================================
# Color Augmentation Functions (IC-Light style)
# ============================================================================
def random_color_jitter(
    image: torch.Tensor,
    mask: torch.Tensor,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.1,
    prob: float = 0.5,
) -> torch.Tensor:
    if random.random() > prob:
        return image
    
    # image: [C, H, W], normalized to [-1, 1]
    # mask: [1, H, W], binary
    img = image.clone()
    
    # Convert to [0, 1] range for processing
    img = img * 0.5 + 0.5
    
    # Brightness
    if brightness > 0:
        factor = 1.0 + random.uniform(-brightness, brightness)
        img = img * factor
    
    # Contrast
    if contrast > 0:
        factor = 1.0 + random.uniform(-contrast, contrast)
        mean = img.mean(dim=[1, 2], keepdim=True)
        img = (img - mean) * factor + mean
    
    # Saturation
    if saturation > 0:
        factor = 1.0 + random.uniform(-saturation, saturation)
        gray = 0.299 * img[0:1] + 0.587 * img[1:2] + 0.114 * img[2:3]
        img = img * factor + gray * (1.0 - factor)
    
    # Hue shift (simplified)
    if hue > 0:
        hue_shift = random.uniform(-hue, hue)
        # Simple hue rotation by channel mixing
        cos_h = math.cos(hue_shift * math.pi)
        sin_h = math.sin(hue_shift * math.pi)
        r, g, b = img[0], img[1], img[2]
        new_r = r * cos_h + g * sin_h * 0.5 - b * sin_h * 0.5
        new_g = g * cos_h + b * sin_h * 0.5 - r * sin_h * 0.5  
        new_b = b * cos_h + r * sin_h * 0.5 - g * sin_h * 0.5
        img = torch.stack([new_r, new_g, new_b], dim=0)
    
    # Clamp and convert back to [-1, 1]
    img = img.clamp(0, 1)
    img = img * 2.0 - 1.0
    
    # Only apply to mask region
    mask_3c = mask.expand(3, -1, -1)
    result = image * (1 - mask_3c) + img * mask_3c
    
    return result


def random_lighting_direction(
    image: torch.Tensor,
    mask: torch.Tensor,
    strength: float = 0.3,
    prob: float = 0.5,
) -> torch.Tensor:
    if random.random() > prob:
        return image
    
    # image: [C, H, W], normalized to [-1, 1]
    # mask: [1, H, W], binary
    C, H, W = image.shape
    
    # Random lighting direction: left, right, top, bottom, or radial
    direction = random.choice(['left', 'right', 'top', 'bottom', 'radial'])
    
    # Create gradient
    if direction == 'left':
        gradient = torch.linspace(1.0 + strength, 1.0 - strength, W).view(1, 1, W).expand(1, H, W)
    elif direction == 'right':
        gradient = torch.linspace(1.0 - strength, 1.0 + strength, W).view(1, 1, W).expand(1, H, W)
    elif direction == 'top':
        gradient = torch.linspace(1.0 + strength, 1.0 - strength, H).view(1, H, 1).expand(1, H, W)
    elif direction == 'bottom':
        gradient = torch.linspace(1.0 - strength, 1.0 + strength, H).view(1, H, 1).expand(1, H, W)
    else:  # radial
        y = torch.linspace(-1, 1, H).view(H, 1)
        x = torch.linspace(-1, 1, W).view(1, W)
        dist = torch.sqrt(x**2 + y**2).unsqueeze(0)
        gradient = 1.0 + strength * (1.0 - dist.clamp(0, 1))

    gradient = gradient.to(image.device)

    # Apply gradient to image (convert to [0,1] first)
    img = image * 0.5 + 0.5
    img = img * gradient
    img = img.clamp(0, 1)
    img = img * 2.0 - 1.0
    
    # Only apply to mask region
    mask_3c = mask.expand(3, -1, -1)
    result = image * (1 - mask_3c) + img * mask_3c
    
    return result


def random_color_temperature(
    image: torch.Tensor,
    mask: torch.Tensor,
    temp_range: Tuple[float, float] = (-0.2, 0.2),
    prob: float = 0.5,
) -> torch.Tensor:
    if random.random() > prob:
        return image
    
    # image: [C, H, W], normalized to [-1, 1]
    temp = random.uniform(temp_range[0], temp_range[1])
    
    img = image.clone()
    # Warm: increase red, decrease blue
    # Cool: decrease red, increase blue
    img[0] = img[0] + temp  # Red
    img[2] = img[2] - temp  # Blue
    img = img.clamp(-1, 1)
    
    # Only apply to mask region
    mask_3c = mask.expand(3, -1, -1)
    result = image * (1 - mask_3c) + img * mask_3c
    
    return result


def apply_iclight_style_augmentation(
    image: torch.Tensor,
    mask: torch.Tensor,
    color_jitter_prob: float = 0.5,
    lighting_prob: float = 0.5,
    temperature_prob: float = 0.5,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.1,
    lighting_strength: float = 0.3,
    temp_range: Tuple[float, float] = (-0.2, 0.2),
) -> torch.Tensor:
    result = image
    
    # Apply color jitter
    result = random_color_jitter(
        result, mask,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        prob=color_jitter_prob,
    )
    
    # Apply lighting direction
    result = random_lighting_direction(
        result, mask,
        strength=lighting_strength,
        prob=lighting_prob,
    )
    
    # Apply color temperature
    result = random_color_temperature(
        result, mask,
        temp_range=temp_range,
        prob=temperature_prob,
    )
    
    return result


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


def read_mask_video(video_path: str, num_frames: int = None) -> Tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open mask video: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale and binarize
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_binary = (frame_gray > 127).astype(np.uint8)
        frames.append(frame_binary)
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames read from mask video: {video_path}")
    
    return np.array(frames), fps


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
    ref_frame_idx: int,
    ref_image_aug: Optional[object] = None,
    ref_mask_aug: Optional[object] = None,
    caption: str = "",
    fps: float = 24,
):
    """
    Save generated training data for debugging/visualization.
    """
    sample_dir = Path(output_dir) / f"sample_{sample_idx:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original video
    save_video_frames([f for f in video], str(sample_dir / "original.mp4"), int(fps))
    
    # Save masked video
    save_video_frames([f for f in masked_video], str(sample_dir / "masked.mp4"), int(fps))
    
    # Save mask video
    mask_rgb = [np.stack([m * 255] * 3, axis=-1).astype(np.uint8) for m in mask_video]
    save_video_frames(mask_rgb, str(sample_dir / "mask.mp4"), int(fps))
    
    # Save first frame mask
    cv2.imwrite(str(sample_dir / "mask_frame0.png"), mask_video[0] * 255)
    
    # Save reference image
    cv2.imwrite(str(sample_dir / "ref_image.png"), cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR))
    
    # Save ref frame index
    with open(sample_dir / "ref_frame_idx.txt", "w") as f:
        f.write(str(ref_frame_idx))

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
    
    return {
        'sample_dir': str(sample_dir),
        'ref_frame_idx': ref_frame_idx,
        'caption': caption,
    }


class MaskVideoInpaintingDataset(Dataset):
    
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
        self.mask_video_root = args.mask_video_root
        self.mask_csv_file = getattr(args, 'mask_csv_file', None)
        # Debug/visualization settings
        self.save_debug = save_debug
        self.debug_output_dir = debug_output_dir
        self.debug_save_prob = debug_save_prob
        self.debug_counter = 0
        
        # Augmentation settings for ref (affine)
        self.ref_aug_degrees = getattr(args, 'ref_aug_degrees', 10.0)
        self.ref_aug_scale_range = getattr(args, 'ref_aug_scale_range', (0.8, 2.0))
        self.ref_aug_hflip_prob = getattr(args, 'ref_aug_hflip_prob', 0.5)
        self.ref_aug_shear_range = getattr(args, 'ref_aug_shear_range', (-10.0, 10.0))
        
        # IC-Light style color augmentation settings
        self.color_jitter_prob = getattr(args, 'color_jitter_prob', 0.5)
        self.lighting_prob = getattr(args, 'lighting_prob', 0.5)
        self.temperature_prob = getattr(args, 'temperature_prob', 0.5)
        
        # IC-Light real relighting (diffusion-based, slower but higher quality)
        self.use_iclight_relight = getattr(args, 'use_iclight_relight', False)
        self.iclight_relight_prob = getattr(args, 'iclight_relight_prob', 0.3)
        self.iclight_steps = getattr(args, 'iclight_steps', 10)
        self._iclight = None
        
        self._load_metadata()
        self._load_mask_metadata()
        print(f'[MaskVideoInpaintingDataset] len of video metadata: {len(self.metadata)}')
        print(f'[MaskVideoInpaintingDataset] len of mask metadata: {len(self.mask_metadata)}')

    def _load_metadata(self):
        self.metadata = []
        for csv_file in self.csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata.append(row)

    def _load_mask_metadata(self):
        self.mask_metadata = []
        if self.mask_csv_file is not None:
            with open(self.mask_csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.mask_metadata.append(row)
        else:
            self.mask_metadata = None

    def _get_random_mask_video_path(self) -> str:
        if self.mask_metadata is not None and len(self.mask_metadata) > 0:
            mask_row = random.choice(self.mask_metadata)
            return mask_row.get('mask_file_path')
        return None

    def _random_shrink_mask(self, mask_video: np.ndarray, shrink_prob: float = 0.5, scale_range: Tuple[float, float] = (0.5, 1.0)) -> np.ndarray:
        if random.random() > shrink_prob:
            return mask_video
        
        scale = random.uniform(scale_range[0], scale_range[1])
        if scale >= 1.0:
            return mask_video
        
        F, H, W = mask_video.shape
        shrunk_masks = []
        
        for i in range(F):
            mask = mask_video[i]
            
            # 找到mask的边界框
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                shrunk_masks.append(mask)
                continue
            
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            
            # 计算中心点
            cy = (y_min + y_max) / 2
            cx = (x_min + x_max) / 2
            
            # 缩小mask区域
            new_h = int((y_max - y_min) * scale)
            new_w = int((x_max - x_min) * scale)
            
            if new_h < 1 or new_w < 1:
                shrunk_masks.append(mask)
                continue
            
            # 裁剪原始mask区域并缩放
            mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
            mask_shrunk = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # 创建新的空mask并放置缩小后的mask到中心位置
            new_mask = np.zeros_like(mask)
            new_y_min = int(cy - new_h / 2)
            new_x_min = int(cx - new_w / 2)
            new_y_max = new_y_min + new_h
            new_x_max = new_x_min + new_w
            
            # 裁剪到图像边界内
            src_y_start = max(0, -new_y_min)
            src_x_start = max(0, -new_x_min)
            dst_y_start = max(0, new_y_min)
            dst_x_start = max(0, new_x_min)
            dst_y_end = min(H, new_y_max)
            dst_x_end = min(W, new_x_max)
            src_y_end = src_y_start + (dst_y_end - dst_y_start)
            src_x_end = src_x_start + (dst_x_end - dst_x_start)
            
            if dst_y_end > dst_y_start and dst_x_end > dst_x_start:
                new_mask[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = mask_shrunk[src_y_start:src_y_end, src_x_start:src_x_end]
            
            shrunk_masks.append(new_mask)
        
        return np.array(shrunk_masks)

    def __len__(self):
        return len(self.metadata) * self.repeat

    def _aug(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame) if transform is not None else frame

    def _find_valid_ref_frame(self, mask_video: np.ndarray) -> int:

        valid_frames = []
        for i in range(len(mask_video)):
            if mask_video[i].sum() > 0:
                valid_frames.append(i)
        
        if len(valid_frames) == 0:
            return 0
        
        return random.choice(valid_frames)

    def _apply_iclight_relight(self, ref_image: torch.Tensor, ref_mask: torch.Tensor) -> torch.Tensor:
        if self._iclight is None:
            self._iclight = get_iclight_relight()
        
        # Convert tensor [C, H, W] normalized [-1, 1] to numpy [H, W, C] uint8
        img = ref_image.clone()
        img = img * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        img = img.clamp(0, 1)
        img = (img * 255).byte()
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        
        # Convert mask tensor [1, H, W] to numpy [H, W]
        mask_np = ref_mask[0].cpu().numpy()
        mask_np = (mask_np > 0.5).astype(np.uint8)
        
        H, W = img_np.shape[:2]
        
        # Prepare foreground (mask region with gray background)
        input_fg = np.ones((H, W, 3), dtype=np.uint8) * 127
        mask_3c = np.stack([mask_np] * 3, axis=-1)
        input_fg[mask_3c > 0.5] = img_np[mask_3c > 0.5]
        
        try:
            # Run IC-Light relighting
            relit = self._iclight.relight(
                input_fg=input_fg,
                prompt="",
                image_width=min(512, W),
                image_height=min(512, H),
                seed=random.randint(0, 2**31 - 1),
                steps=self.iclight_steps,
                cfg=7.0,
                use_random_bg=True,
            )
            
            # Resize back to original size if needed
            if relit.shape[0] != H or relit.shape[1] != W:
                relit = cv2.resize(relit, (W, H), interpolation=cv2.INTER_LANCZOS4)
            
            # Apply relit result only to mask region
            result_np = img_np.copy()
            result_np[mask_3c > 0.5] = relit[mask_3c > 0.5]
            
            # Convert back to tensor [-1, 1]
            result = torch.from_numpy(result_np).float() / 255.0
            result = result.permute(2, 0, 1)  # [C, H, W]
            result = result * 2.0 - 1.0  # [0, 1] -> [-1, 1]
            
            return result
        except Exception as e:
            print(f"[ICLight] Relighting failed: {e}, using original image")
            return ref_image

    def __getitem__(self, index):
        while True:
            index = index % len(self.metadata)
            
            # Read metadata
            raw_data = self.metadata[index]
            vid_path = os.path.join(self.data_root, raw_data.get('video_path', raw_data.get('vid_path', '')))
            
            # Get mask_video_path: from separate CSV (random) or from same CSV
            if self.mask_metadata is not None:
                random_mask_path = self._get_random_mask_video_path()
                mask_vid_path = os.path.join(self.mask_video_root, random_mask_path)
            else:
                mask_vid_path = os.path.join(self.mask_video_root, raw_data.get('mask_file_path'))
            
            vid_caption = raw_data.get('prompt', raw_data.get('caption', ''))

            try:
                # Load original video
                video_reader = VideoReader(vid_path)
                
                # Load mask video
                mask_video_full, mask_fps = read_mask_video(mask_vid_path)
                
                # Handle frame count mismatch
                video_len = len(video_reader)
                mask_len = len(mask_video_full)
                
                # Use minimum length
                available_frames = min(video_len, mask_len)
                
                if available_frames < self.nframes:
                    # Handle short videos by repeating frames
                    repeat_num = math.ceil(self.nframes / available_frames)
                    if random.random() >= 0.5:
                        temp_list = list(range(available_frames)) + list(range(available_frames))[::-1][1:-1]
                        all_frames = temp_list * repeat_num
                    else:
                        all_frames = list(range(available_frames)) + [available_frames - 1] * (self.nframes - available_frames + 3)
                else:
                    all_frames = list(range(available_frames))

                # Select random clip
                rand_idx = random.randint(0, max(0, len(all_frames) - self.nframes - 1))
                frame_indices = all_frames[rand_idx:rand_idx + self.nframes]
                
                if len(frame_indices) < self.nframes:
                    print(f"vid frames are {len(frame_indices)}")
                    index += 1
                    continue
                
                # Read video frames
                video = video_reader.get_batch(frame_indices).asnumpy()  # [F, H, W, C]
                fps = video_reader.get_avg_fps()
                
                # Get corresponding mask frames
                mask_video = np.array([mask_video_full[i] for i in frame_indices])  # [F, H, W]
                
                break
                
            except Exception as e:
                print(f"Load data failed! video={vid_path}, mask={mask_vid_path}, error={e}")
                index += 1
                continue

        assert video.shape[0] == self.nframes, f'{video.shape[0]}, self.nframes={self.nframes}'
        assert mask_video.shape[0] == self.nframes, f'mask frames: {mask_video.shape[0]}'

        height_v, width_v, _ = video[0].shape
        ori_ratio = height_v / width_v

        # Resize mask to match video size if needed
        if mask_video.shape[1] != height_v or mask_video.shape[2] != width_v:
            mask_video_resized = []
            for m in mask_video:
                m_resized = cv2.resize(m, (width_v, height_v), interpolation=cv2.INTER_NEAREST)
                mask_video_resized.append(m_resized)
            mask_video = np.array(mask_video_resized)

        # 随机缩小mask (50%概率，缩放0.5-1.0)
        mask_video = self._random_shrink_mask(mask_video, shrink_prob=0.5, scale_range=(0.5, 1.0))

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
        mask_video_tensors = []
        masked_video_np = []  # For debug saving
        
        state = torch.get_rng_state()
        
        for t in range(self.nframes):
            frame = video[t]
            mask_frame = mask_video[t]
            
            # Target image (ground truth)
            img = Image.fromarray(frame)
            target_img = self._aug(img, img_transform, state)
            target_images.append(target_img)
            
            # Masked video (mask==1 region blacked out)
            masked_frame = frame.copy()
            masked_frame[mask_frame > 0.5] = 0  # Black out mask==1 region
            masked_video_np.append(masked_frame)
            masked_img = Image.fromarray(masked_frame)
            masked_img = self._aug(masked_img, img_transform, state)
            masked_video_frames.append(masked_img)
            
            # Process mask for this frame
            mask_pil = Image.fromarray((mask_frame * 255).astype(np.uint8)).convert("L")
            mask_tensor = self._aug(mask_pil, mask_transform, state)
            mask_video_tensors.append(mask_tensor)

        # Stack tensors
        target_images = torch.stack(target_images, dim=0)  # [F, C, H, W]
        masked_video = torch.stack(masked_video_frames, dim=0)  # [F, C, H, W]
        mask_video_tensor = torch.stack(mask_video_tensors, dim=0)  # [F, 1, H, W]
        
        # ============================================
        # Select reference frame (random frame with mask)
        # ============================================
        ref_frame_idx = self._find_valid_ref_frame(mask_video)
        
        # Extract reference image (selected frame with mask==1 kept)
        ref_frame = video[ref_frame_idx]
        ref_mask = mask_video[ref_frame_idx]
        
        ref_frame_fg = ref_frame.copy()
        ref_frame_fg[ref_mask < 0.5] = 0  # Keep only mask==1 region
        ref_frame_fg_pil = Image.fromarray(ref_frame_fg)
        ref_image = self._aug(ref_frame_fg_pil, img_transform, state)  # [C, H, W]
        
        # Reference mask (same frame's mask)
        ref_mask_pil = Image.fromarray((ref_mask * 255).astype(np.uint8)).convert("L")
        ref_masked_image = self._aug(ref_mask_pil, mask_transform, state)  # [1, H, W]
        
        mask_image = mask_video_tensor[0]  # [1, H, W]
        ref_image_aug, ref_masked_image_aug, _ = random_affine_preserve_mask(
            ref_image,
            ref_masked_image,
            degrees=self.ref_aug_degrees,
            scale_range=self.ref_aug_scale_range,
            hflip_prob=self.ref_aug_hflip_prob,
            shear_range=self.ref_aug_shear_range,
        )
        
        # Apply IC-Light real relighting or simple color augmentation
        if self.use_iclight_relight and random.random() < self.iclight_relight_prob:
            ref_image_aug = self._apply_iclight_relight(ref_image_aug, ref_masked_image_aug)
        else:
            ref_image_aug = apply_iclight_style_augmentation(
                ref_image_aug,
                ref_masked_image_aug,
                color_jitter_prob=self.color_jitter_prob,
                lighting_prob=self.lighting_prob,
                temperature_prob=self.temperature_prob,
            )

        # Debug saving
        if self.save_debug and random.random() < self.debug_save_prob:
            save_debug_outputs(
                output_dir=self.debug_output_dir,
                sample_idx=self.debug_counter,
                video=video,
                masked_video=np.stack(masked_video_np, axis=0),
                mask_video=mask_video,
                ref_image=ref_frame_fg,
                ref_frame_idx=ref_frame_idx,
                ref_image_aug=ref_image_aug,
                ref_mask_aug=ref_masked_image_aug,
                caption=vid_caption,
                fps=fps,
            )
            self.debug_counter += 1
        
        outputs = {
            'target_images': target_images,           # [F, C, H, W]
            'masked_video': masked_video,             # [F, C, H, W]
            'mask_video': mask_video_tensor,          # [F, 1, H, W]
            'mask_image': mask_image,                 # [1, H, W]
            'ref_image': ref_image_aug,               # [C, H, W] - augmented
            'ref_masked_image': ref_masked_image_aug, # [1, H, W] - binary mask, augmented
            'ref_frame_idx': ref_frame_idx,           # int - which frame was used as ref
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs


# ============================================================================
# Debug Script Entry Point
# ============================================================================

def debug_dataset(
    csv_file: str,
    data_root: str,
    mask_csv_file: str = None,
    mask_video_root: str = None,
    output_dir: str = "./debug_outputs",
    num_samples: int = 5,
    nframes: int = 81,
):
    from dataclasses import dataclass
    from tqdm import tqdm
    
    @dataclass
    class DebugArgs:
        csv_file_list: List[str]
        data_root: str
        mask_csv_file: str
        mask_video_root: str
        nframes: int
        ref_aug_degrees: float = 10.0
        ref_aug_scale_range: Tuple[float, float] = (0.8, 2.0)
        ref_aug_hflip_prob: float = 0.5
        ref_aug_shear_range: Tuple[float, float] = (-10.0, 10.0)
    
    args = DebugArgs(
        csv_file_list=[csv_file],
        data_root=data_root,
        mask_csv_file=mask_csv_file,
        mask_video_root=mask_video_root or data_root,
        nframes=nframes,
    )
    
    print(f"Using MaskVideoInpaintingDataset")
    print(f"  video_csv: {csv_file}")
    print(f"  data_root: {data_root}")
    print(f"  mask_csv_file: {mask_csv_file}")
    print(f"  mask_video_root: {mask_video_root or data_root}")
    
    dataset = MaskVideoInpaintingDataset(
        args,
        save_debug=True,
        debug_output_dir=output_dir,
        debug_save_prob=1.0,
    )
    
    actual_samples = min(num_samples, len(dataset))
    print(f"Dataset length: {len(dataset)}")
    print(f"Processing {actual_samples} samples")
    print(f"Output directory: {output_dir}")
    
    for i in tqdm(range(actual_samples), desc="Processing samples"):
        try:
            sample = dataset[i]
            print(f"  Sample {i}: ref_frame_idx={sample['ref_frame_idx']}, "
                  f"target={sample['target_images'].shape}, "
                  f"mask_video={sample['mask_video'].shape}")
        except Exception as e:
            print(f"  Sample {i} FAILED: {e}")
    
    print(f"\nDone! Check outputs in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug MaskVideoInpaintingDataset")
    parser.add_argument("--csv_file", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/data/wanErase_data/train_data_filtered_by_sft.csv", help="Path to video CSV file (video_path, caption)")
    parser.add_argument("--data_root", type=str, default="/", help="Root directory for videos")
    parser.add_argument("--mask_csv_file", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/data/wanErase_data/mask_paths.csv", help="Path to mask CSV file (mask_video_path), randomly selected")
    parser.add_argument("--mask_video_root", type=str, default="/", help="Root directory for mask videos")
    parser.add_argument("--output_dir", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/sft/", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to save")
    parser.add_argument("--nframes", type=int, default=81, help="Number of frames")
    
    args = parser.parse_args()
    
    debug_dataset(
        csv_file=args.csv_file,
        data_root=args.data_root,
        mask_csv_file=args.mask_csv_file,
        mask_video_root=args.mask_video_root,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        nframes=args.nframes,
    )
