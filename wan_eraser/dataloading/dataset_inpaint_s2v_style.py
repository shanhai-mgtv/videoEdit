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
from saber_mask import generate_mask


# ============================================================================
# Aspect Ratio Table
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
# Mask Region Utilities
# ============================================================================

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


def crop_and_pad_to_maximize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
    padding_value: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop foreground region and pad to target size WITHOUT stretching.
    
    This function:
    1. Finds the bounding box of the mask region
    2. Crops the image/mask to that region
    3. Scales to fit within target_size while maintaining aspect ratio
    4. Pads to target_size (centered)
    
    Args:
        image: [H, W, C] input image
        mask: [H, W] binary mask
        target_size: (target_H, target_W)
        padding_value: value to use for padding
    
    Returns:
        padded_image: [target_H, target_W, C]
        padded_mask: [target_H, target_W]
    """
    target_h, target_w = target_size
    
    # Get bounding box of mask region
    y_min, y_max, x_min, x_max = get_mask_bbox(mask, padding=5)
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    
    # Crop to bounding box
    cropped_image = image[y_min:y_max+1, x_min:x_max+1].copy()
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1].copy()
    
    # Calculate scale to fit within target while maintaining aspect ratio
    scale_h = target_h / bbox_h
    scale_w = target_w / bbox_w
    scale = min(scale_h, scale_w)  # Use smaller scale to fit
    
    # Resize maintaining aspect ratio
    new_h = int(bbox_h * scale)
    new_w = int(bbox_w * scale)
    
    # Ensure dimensions are at least 1
    new_h = max(1, new_h)
    new_w = max(1, new_w)
    
    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Create padded output (centered)
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    
    # Pad image
    if len(image.shape) == 3:
        padded_image = np.full((target_h, target_w, image.shape[2]), padding_value, dtype=image.dtype)
        padded_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image
    else:
        padded_image = np.full((target_h, target_w), padding_value, dtype=image.dtype)
        padded_image[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_image
    
    # Pad mask
    padded_mask = np.zeros((target_h, target_w), dtype=mask.dtype)
    padded_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized_mask
    
    return padded_image, padded_mask


def extract_reference_with_augmentation(
    frame: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
    enable_geometric_aug: bool = True,
    rotation_range: Tuple[float, float] = (-15, 15),  # degrees
    scale_range: Tuple[float, float] = (0.9, 1.0),    # relative to max fit scale
    translate_range: Tuple[float, float] = (-0.05, 0.05),  # fraction of size
) -> np.ndarray:
    """
    Extract foreground (mask==1 region) from frame with geometric augmentation.
    Maximizes the mask region in the output while applying augmentation.
    
    Args:
        frame: [H, W, C] input frame
        mask: [H, W] binary mask
        target_size: (target_H, target_W)
        enable_geometric_aug: whether to apply geometric augmentation
        rotation_range: (min_deg, max_deg) for rotation
        scale_range: (min_scale, max_scale) relative to max-fit scale
        translate_range: (min_frac, max_frac) for translation as fraction of size
    
    Returns:
        ref_image: [target_H, target_W, C] reference image with foreground maximized
    """
    target_h, target_w = target_size
    
    # Black out background
    fg_frame = frame.copy()
    fg_frame[mask < 0.5] = 0
    
    # Get bounding box of mask region
    y_min, y_max, x_min, x_max = get_mask_bbox(mask, padding=5)
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    
    # Crop to bounding box (maximize mask region)
    cropped_image = fg_frame[y_min:y_max+1, x_min:x_max+1].copy()
    
    # Calculate MAX scale to fit within target while maintaining aspect ratio
    scale_h = target_h / bbox_h
    scale_w = target_w / bbox_w
    max_fit_scale = min(scale_h, scale_w)  # This maximizes the object in frame
    
    if enable_geometric_aug:
        # Random geometric augmentation (relative to max fit)
        aug_scale = random.uniform(scale_range[0], scale_range[1])
        aug_rotation = random.uniform(rotation_range[0], rotation_range[1])
        aug_tx = random.uniform(translate_range[0], translate_range[1])
        aug_ty = random.uniform(translate_range[0], translate_range[1])
    else:
        aug_scale = 1.0
        aug_rotation = 0.0
        aug_tx = 0.0
        aug_ty = 0.0
    
    # Final scale: maximize then apply augmentation
    final_scale = max_fit_scale * aug_scale
    
    # Resize maintaining aspect ratio
    new_h = int(bbox_h * final_scale)
    new_w = int(bbox_w * final_scale)
    new_h = max(1, min(new_h, target_h))
    new_w = max(1, min(new_w, target_w))
    
    resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Apply rotation around center
    if abs(aug_rotation) > 0.1:
        # Expand canvas for rotation to avoid clipping
        diag = int(np.ceil(np.sqrt(new_h**2 + new_w**2)))
        pad_h = (diag - new_h) // 2
        pad_w = (diag - new_w) // 2
        
        padded = np.zeros((diag, diag, 3), dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized_image
        
        center = (diag // 2, diag // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, aug_rotation, 1.0)
        rotated = cv2.warpAffine(
            padded, rot_matrix, (diag, diag),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        
        # Crop back to original size from center
        start_h = (diag - new_h) // 2
        start_w = (diag - new_w) // 2
        resized_image = rotated[start_h:start_h+new_h, start_w:start_w+new_w]
    
    # Create output canvas
    ref_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate placement position (centered + translation offset)
    base_y = (target_h - new_h) // 2
    base_x = (target_w - new_w) // 2
    
    # Apply translation offset (small offset to maintain maximized appearance)
    offset_y = int(aug_ty * new_h)  # Offset relative to object size, not canvas
    offset_x = int(aug_tx * new_w)
    
    paste_y = max(0, min(target_h - new_h, base_y + offset_y))
    paste_x = max(0, min(target_w - new_w, base_x + offset_x))
    
    # Paste resized image onto canvas
    ref_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized_image
    
    return ref_image


# ============================================================================
# Debug Utilities
# ============================================================================

def save_video_frames(frames: List[np.ndarray], output_path: str, fps: int = 24):
    """Save list of numpy frames as video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", bitrate="10M", logger=None)


def save_debug_outputs_s2v_style(
    output_dir: str,
    sample_idx: int,
    video: np.ndarray,
    masked_video: np.ndarray,
    mask: np.ndarray,
    ref_image: np.ndarray,
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
    
    return {
        'sample_dir': str(sample_dir),
        'mask_path': str(sample_dir / "mask.png"),
        'ref_image_path': str(sample_dir / "ref_image.png"),
        'caption': caption,
    }


# ============================================================================
# S2V-Style Dataset
# ============================================================================

class InpaintingDatasetS2VStyle(Dataset):
    """
    S2V-style dataset for video inpainting training.
    
    Key features:
    - No reference augmentation (ref image is clean foreground)
    - No reference mask output
    - Reference image is cropped and padded to maximize mask region (no stretching)
    
    Args:
        args: config with csv_file_list, data_root, nframes, etc.
        save_debug: whether to save debug outputs
        debug_output_dir: directory for debug outputs
        debug_save_prob: probability of saving each sample
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
        print(f'[InpaintingDatasetS2VStyle] len of metadata: {len(self.metadata)}')

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
        target_h, target_w = closest_size
        
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

        # Define transforms for video frames
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
        
        # ============================================
        # Extract reference image with geometric + color augmentation
        # - Rotation, scale, translation to make ref different from original
        # - Prevents model from simply copying
        # ============================================
        first_frame = video[0]
        ref_image_np = extract_reference_with_augmentation(
            first_frame, mask, (target_h, target_w),
            enable_geometric_aug=True,
            rotation_range=(-15, 15),    
            scale_range=(0.85, 1.15),   
            translate_range=(-0.1, 0.1),  
        )
        
        # Convert to tensor
        ref_image_pil = Image.fromarray(ref_image_np)
        
        # Light augmentation: color jitter to make ref different from original
        ref_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1, 
                contrast=0.1,    
                saturation=0.1,  
                hue=0.02,       
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        ref_image = ref_transform(ref_image_pil)  # [C, H, W]
        
        mask_image = mask_video[0]  # [1, H, W]

        # Debug saving
        if self.save_debug and random.random() < self.debug_save_prob:
            save_debug_outputs_s2v_style(
                output_dir=self.debug_output_dir,
                sample_idx=self.debug_counter,
                video=video,
                masked_video=np.stack(masked_video_np, axis=0),
                mask=mask,
                ref_image=ref_image_np,
                caption=vid_caption,
                fps=fps,
            )
            self.debug_counter += 1
        
        outputs = {
            'target_images': target_images,      # [F, C, H, W]
            'masked_video': masked_video,        # [F, C, H, W]
            'mask_video': mask_video,            # [F, 1, H, W]
            'mask_image': mask_image,            # [1, H, W]
            'ref_image': ref_image,             
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs
