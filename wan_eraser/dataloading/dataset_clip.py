"""
Dataset for CLIP-based video inpainting training.
Key features:
1. No mask augmentation - uses consistent mask across frames
2. ref_image selection with minimal mask area (minimal black frames)
3. Returns ref_image for CLIP encoding
4. Supports mask video path input (like dataset_inpaint_with_mask_video.py)
"""
import random
import math
import csv
import os
from typing import Optional, Tuple, List

import cv2
import torch
import numpy as np
from PIL import Image
from decord import VideoReader
from torch.utils.data import Dataset
from torchvision import transforms


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


def read_mask_video(video_path: str) -> Tuple[np.ndarray, float]:
    """Read mask video and return binary mask frames."""
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


class CLIPDataset(Dataset):
    """
    Dataset for CLIP-based video inpainting training.
    - No mask augmentation
    - ref_image is selected as the frame with smallest mask area (minimal black frames)
    - Supports mask video path from CSV
    """
    def __init__(self, args):
        self.args = args
        self.repeat = 1
        self.nframes = args.nframes
        self.csv_file_list = args.csv_file_list
        self.data_root = args.data_root
        self.mask_video_root = getattr(args, 'mask_video_root', args.data_root)
        self.mask_csv_file = getattr(args, 'mask_csv_file', None)
        
        self._load_metadata()
        self._load_mask_metadata()
        
        print(f'[CLIPDataset] len of video metadata: {len(self.metadata)}')
        print(f'[CLIPDataset] len of mask metadata: {len(self.mask_metadata) if self.mask_metadata else 0}')

    def _load_metadata(self):
        self.metadata = []
        for csv_file in self.csv_file_list:
            with open(csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.metadata.append(row)

    def _load_mask_metadata(self):
        """Load mask video paths from separate CSV file."""
        self.mask_metadata = []
        if self.mask_csv_file is not None:
            with open(self.mask_csv_file, 'r', encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    self.mask_metadata.append(row)
        else:
            self.mask_metadata = None

    def _get_random_mask_video_path(self) -> str:
        """Get random mask video path from mask CSV."""
        if self.mask_metadata is not None and len(self.mask_metadata) > 0:
            mask_row = random.choice(self.mask_metadata)
            return mask_row.get('mask_file_path', mask_row.get('mask_path', ''))
        return None
    
    def __len__(self):
        return len(self.metadata) * self.repeat

    def _aug(self, frame, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(frame) if transform is not None else frame

    def _find_valid_ref_frame(self, mask_video: np.ndarray) -> int:
        """
        Find a random frame with valid mask (mask area > 0).
        Returns random frame index from valid frames.
        """
        valid_frames = []
        for i in range(len(mask_video)):
            if mask_video[i].sum() > 0:
                valid_frames.append(i)
        
        if len(valid_frames) == 0:
            return 0
        
        return random.choice(valid_frames)

    def _extract_and_scale_ref(self, frame: np.ndarray, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Extract mask region from frame and scale to minimize black borders.
        
        Args:
            frame: [H, W, C] original video frame
            mask: [H, W] binary mask (1=foreground)
            target_size: (H, W) target output size
        
        Returns:
            ref_image: [H, W, C] scaled reference image with minimal black borders
        """
        H, W, C = frame.shape
        target_H, target_W = target_size
        
        # Extract foreground only (mask==1 region)
        fg_frame = frame.copy()
        fg_frame[mask < 0.5] = 0  # Black out background
        
        # Find bounding box of mask region
        ys, xs = np.where(mask > 0.5)
        if len(ys) == 0 or len(xs) == 0:
            # No mask, return original frame
            return cv2.resize(fg_frame, (target_W, target_H), interpolation=cv2.INTER_LANCZOS4)
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        # Add small padding
        pad = 5
        y_min = max(0, y_min - pad)
        y_max = min(H - 1, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(W - 1, x_max + pad)
        
        # Crop to bounding box
        cropped = fg_frame[y_min:y_max+1, x_min:x_max+1]
        crop_h, crop_w = cropped.shape[:2]
        
        # Calculate scale to fit target size while minimizing black borders
        scale_h = target_H / crop_h
        scale_w = target_W / crop_w
        scale = min(scale_h, scale_w)  # Use smaller scale to fit
        
        # Scale up the cropped region
        new_h = int(crop_h * scale)
        new_w = int(crop_w * scale)
        scaled = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center in target canvas
        ref_image = np.zeros((target_H, target_W, C), dtype=np.uint8)
        start_y = (target_H - new_h) // 2
        start_x = (target_W - new_w) // 2
        ref_image[start_y:start_y+new_h, start_x:start_x+new_w] = scaled
        
        return ref_image

    def _simple_augment_ref(self, image: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
        """
        Apply simple augmentation to reference image.
        
        Args:
            image: [C, H, W] tensor in range [-1, 1]
            prob: probability of applying augmentation
        
        Returns:
            augmented image tensor
        """
        if random.random() > prob:
            return image
        
        img = image.clone()
        
        # Random horizontal flip
        if random.random() > 0.5:
            img = torch.flip(img, dims=[2])
        
        # Simple brightness/contrast adjustment
        if random.random() > 0.5:
            # Convert to [0, 1]
            img = img * 0.5 + 0.5
            
            # Brightness
            brightness = random.uniform(0.8, 1.2)
            img = img * brightness
            
            # Contrast
            contrast = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = (img - mean) * contrast + mean
            
            # Clamp and convert back to [-1, 1]
            img = img.clamp(0, 1)
            img = img * 2.0 - 1.0
        
        return img

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
                mask_vid_path = os.path.join(self.mask_video_root, raw_data.get('mask_file_path', raw_data.get('mask_path', '')))
            
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

        # Ref transform: already at target size from _extract_and_scale_ref, only normalize
        ref_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Process frames with same random crop for all
        target_images = []
        input_masked_imgs = []
        input_masks = []
        
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
            masked_img = Image.fromarray(masked_frame)
            masked_img = self._aug(masked_img, img_transform, state)
            input_masked_imgs.append(masked_img)
            
            # Process mask for this frame (no augmentation - consistent)
            mask_pil = Image.fromarray((mask_frame * 255).astype(np.uint8)).convert("L")
            mask_tensor = self._aug(mask_pil, mask_transform, state)
            input_masks.append(mask_tensor)

        # Stack tensors
        target_images = torch.stack(target_images, dim=0)  # [F, C, H, W]
        input_masked_imgs = torch.stack(input_masked_imgs, dim=0)  # [F, C, H, W]
        input_masks = torch.stack(input_masks, dim=0)  # [F, 1, H, W]
        
        # Select random valid frame for reference
        ref_frame_idx = self._find_valid_ref_frame(mask_video)
        ref_frame = video[ref_frame_idx]
        ref_mask = mask_video[ref_frame_idx]
        
        # Extract mask region and scale to minimize black borders
        ref_scaled = self._extract_and_scale_ref(ref_frame, ref_mask, tuple(closest_size))
        ref_img = Image.fromarray(ref_scaled)
        
        ref_image = ref_transform(ref_img)  # [C, H, W]
        
        ref_image = self._simple_augment_ref(ref_image, prob=0.5)

        outputs = {
            'target_images': target_images,       # [F, C, H, W]
            'input_masked_imgs': input_masked_imgs,  # [F, C, H, W]
            'input_masks': input_masks,           # [F, 1, H, W]
            'ref_image': ref_image,               # [C, H, W] - for CLIP encoding
            'caption': vid_caption,
            'fps': fps,
        }

        return outputs
