
import os
import sys
import argparse
from typing import Optional, Tuple, List

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from decord import VideoReader
from einops import rearrange
from transformers import UMT5EncoderModel, AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm
from moviepy import ImageSequenceClip

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_wan_s2v_style import WanTransformer3DModelS2VStyle
from models.autoencoder_kl_wan import AutoencoderKLWan
from models.flow_match import FlowMatchScheduler
from pipelines.pipeline_wan_inpainting_s2v_style import (
    WanPipelineS2VStyle, 
    retrieve_latents, 
    prompt_clean
)


# ============================================================================
# Utility Functions
# ============================================================================

def read_video_cv2(video_path: str) -> Tuple[np.ndarray, float]:
    """Read video and return frames and fps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return np.array(frames), fps


def read_mask_image(mask_path: str) -> np.ndarray:
    """Read mask image and convert to binary."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    mask = (mask > 127).astype(np.uint8)
    return mask


def read_ref_image(ref_path: str) -> np.ndarray:
    """Read reference image."""
    ref = cv2.imread(ref_path)
    if ref is None:
        raise ValueError(f"Could not read ref image: {ref_path}")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    return ref


def save_video(frames: np.ndarray, output_path: str, fps: float):
    """Save video frames to file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_list = [f for f in frames]
    clip = ImageSequenceClip(frames_list, fps=fps)
    clip.write_videofile(output_path, codec="libx264", bitrate="10M", logger=None)
    print(f"Saved video to {output_path}")


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


def crop_and_pad_reference(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Crop foreground region and pad to target size WITHOUT stretching.
    """
    target_h, target_w = target_size
    
    # Get bounding box of mask region
    y_min, y_max, x_min, x_max = get_mask_bbox(mask, padding=5)
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1
    
    # Crop to bounding box
    cropped = image[y_min:y_max+1, x_min:x_max+1].copy()
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1].copy()
    
    # Black out background
    cropped[cropped_mask < 0.5] = 0
    
    # Calculate scale to fit within target while maintaining aspect ratio
    scale_h = target_h / bbox_h
    scale_w = target_w / bbox_w
    scale = min(scale_h, scale_w)
    
    new_h = max(1, int(bbox_h * scale))
    new_w = max(1, int(bbox_w * scale))
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size (centered)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
    
    return padded


def prepare_latents(
    vae: AutoencoderKLWan,
    image_or_video: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode image or video to latent space."""
    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, -1, 1, 1, 1)
        .to(device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(
        device, vae.dtype
    )
    latents = retrieve_latents(vae.encode(image_or_video), generator, sample_mode="sample")
    latents = (latents - latents_mean) * latents_std
    return latents.to(dtype=dtype)


def prepare_mask_latents(
    mask: torch.Tensor,
    num_frames: int,
    vae_temporal_scale: int = 4,
    vae_spatial_scale: int = 16,
    num_channels: int = 4,
) -> torch.Tensor:
    """
    Prepare mask at latent resolution.
    
    Args:
        mask: [B, 1, H, W] binary mask
        num_frames: number of video frames
    
    Returns:
        mask_latents: [B, num_channels, F_lat, H_lat, W_lat]
    """
    B, _, H, W = mask.shape
    latent_h = H // vae_spatial_scale
    latent_w = W // vae_spatial_scale
    num_latent_frames = (num_frames - 1) // vae_temporal_scale + 1
    
    # Downsample to latent spatial size
    mask_lat = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
    
    # Expand to video and repeat to num_channels (4 for Wan2.2)
    mask_lat = mask_lat.unsqueeze(2).repeat(1, num_channels, num_latent_frames, 1, 1)
    
    return mask_lat


# ============================================================================
# LoRA Merging
# ============================================================================

def merge_lora_weights(
    transformer: WanTransformer3DModelS2VStyle,
    lora_state_dict: dict,
    alpha: float = 1.0,
) -> WanTransformer3DModelS2VStyle:
    """Merge LoRA weights into transformer."""
    param_dict = dict(transformer.named_parameters())
    merged = []
    skipped = []
    
    for key, lora_A in lora_state_dict.items():
        if not key.endswith("lora_A.weight"):
            continue
        
        base_key = key[:-len(".lora_A.weight")]
        lora_B_key = base_key + ".lora_B.weight"
        
        if lora_B_key not in lora_state_dict:
            skipped.append((base_key, "missing B"))
            continue
        
        lora_B = lora_state_dict[lora_B_key]
        
        # Handle patch_embedding specially
        if "patch_embedding" in base_key:
            target_weight = None
            # Extract the actual embedding name (e.g., "patch_embedding" or "ref_patch_embedding")
            if "ref_patch_embedding" in base_key:
                embedding_name = "ref_patch_embedding"
            else:
                embedding_name = "patch_embedding"
            
            for pname, pval in param_dict.items():
                # Match exact embedding name to avoid confusing patch_embedding with ref_patch_embedding
                if pname.endswith(f"{embedding_name}.weight"):
                    target_weight = pval
                    break
            
            if target_weight is None:
                skipped.append((base_key, "target weight not found"))
                continue
            
            rank = lora_A.shape[0]
            A_flat = lora_A.flatten(1)
            B_flat = lora_B.view(lora_B.shape[0], lora_B.shape[1])
            update = (B_flat @ A_flat).view_as(target_weight) * (alpha / rank)
            target_weight.data += update.to(target_weight.device, dtype=target_weight.dtype)
            merged.append(base_key)
        else:
            target_key = base_key + ".weight"
            target_weight = param_dict.get(target_key, None)
            if target_weight is None:
                skipped.append((base_key, "target weight not found"))
                continue
            
            rank = lora_A.shape[0]
            update = (lora_B @ lora_A) * (alpha / rank)
            target_weight.data += update.to(target_weight.device, dtype=target_weight.dtype)
            merged.append(base_key)
    
    print(f"[LoRA] Merged {len(merged)} layers, skipped {len(skipped)}")
    return transformer


# ============================================================================
# Main Inference Function
# ============================================================================

def infer(
    video_path: str,
    mask_path: str,
    ref_path: str,
    output_path: str,
    prompt: str,
    base_model_path: str,
    transformer_path: str,
    lora_path: Optional[str] = None,
    negative_prompt: str = "",
    infer_h: int = 480,
    infer_w: int = 640,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Run S2V-style video inpainting inference.
    """
    print("=" * 60)
    print("S2V-Style Video Inpainting Inference")
    print("=" * 60)
    
    device = torch.device(device)
    load_dtype = torch.bfloat16
    
    # Load models
    print("Loading models...")
    
    vae = AutoencoderKLWan.from_pretrained(
        base_model_path,
        subfolder="vae",
        torch_dtype=load_dtype,
    ).to(device)
    
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_path,
        subfolder="text_encoder",
        torch_dtype=load_dtype,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        subfolder="tokenizer",
    )
    
    # Load transformer
    transformer = WanTransformer3DModelS2VStyle.from_pretrained(
        transformer_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        in_channels=100,  # patch_embedding: 100ch
        out_channels=48,  # Wan2.2 VAE has 48 latent channels
        fixed_ref_position=30,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    
    # Load LoRA if provided
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}")
        
        # Determine checkpoint directory
        if os.path.isdir(lora_path):
            checkpoint_dir = lora_path
            lora_file = os.path.join(lora_path, "inpaint_s2v_style.safetensors")
            if not os.path.exists(lora_file):
                for f in os.listdir(lora_path):
                    if f.endswith(".safetensors") and f != "s2v_new_layers.safetensors":
                        lora_file = os.path.join(lora_path, f)
                        break
        else:
            checkpoint_dir = os.path.dirname(lora_path)
            lora_file = lora_path
        
        transformer = transformer.to(device=device, dtype=load_dtype)
        
        # Load LoRA weights
        if os.path.exists(lora_file):
            lora_state_dict = load_file(lora_file)
            lora_state_dict = {k.replace("transformer.base_model.model.", ""): v.to(device=device, dtype=load_dtype) 
                              for k, v in lora_state_dict.items()}
            transformer = merge_lora_weights(transformer, lora_state_dict, alpha=1.0)
        else:
            print(f"WARNING: LoRA file not found at {lora_file}")
        
        # Load new layers (ref_patch_embedding, trainable_cond_mask)
        new_layers_file = os.path.join(checkpoint_dir, "s2v_new_layers.safetensors")
        if os.path.exists(new_layers_file):
            print(f"Loading new layers from {new_layers_file}")
            new_layers_dict = load_file(new_layers_file)
            if "ref_patch_embedding.weight" in new_layers_dict:
                transformer.ref_patch_embedding.weight.data.copy_(
                    new_layers_dict["ref_patch_embedding.weight"].to(device=device, dtype=load_dtype))
            if "ref_patch_embedding.bias" in new_layers_dict:
                transformer.ref_patch_embedding.bias.data.copy_(
                    new_layers_dict["ref_patch_embedding.bias"].to(device=device, dtype=load_dtype))
            if "trainable_cond_mask.weight" in new_layers_dict:
                transformer.trainable_cond_mask.weight.data.copy_(
                    new_layers_dict["trainable_cond_mask.weight"].to(device=device, dtype=load_dtype))
            print("Loaded new layers (ref_patch_embedding, trainable_cond_mask)")
        else:
            print(f"WARNING: New layers file not found at {new_layers_file}")
    
    transformer = transformer.to(device=device, dtype=load_dtype)
    
    # Create pipeline
    scheduler = FlowMatchScheduler(shift=7, sigma_min=0.0, extra_one_step=True)
    pipeline = WanPipelineS2VStyle(
        tokenizer=tokenizer,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("=" * 60)
    print("Loading input data...")
    print("=" * 60)
    
    # Read inputs
    video, fps = read_video_cv2(video_path)
    mask = read_mask_image(mask_path)
    ref_image = read_ref_image(ref_path)
    
    num_frames = len(video)
    orig_H, orig_W = video[0].shape[:2]
    
    # Use infer_h and infer_w for inference
    H, W = infer_h, infer_w
    print(f"Original video: {num_frames} frames, {orig_H}x{orig_W}, {fps:.2f} fps")
    print(f"Inference size: {H}x{W}")
    
    # Resize video to inference size
    video = np.array([cv2.resize(frame, (W, H)) for frame in video])
    
    print(f"Video: {num_frames} frames, {H}x{W}, {fps:.2f} fps")
    print(f"Mask: {mask.shape}")
    print(f"Ref image: {ref_image.shape}")
    
    # Resize mask if needed
    if mask.shape[0] != H or mask.shape[1] != W:
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        print(f"Resized mask to {mask.shape}")
    
    # Prepare reference image (crop foreground, pad to maximize mask region)
    ref_processed = crop_and_pad_reference(ref_image, mask, (H, W))
    
    # Prepare tensors
    # Video: [F, H, W, C] -> [1, C, F, H, W]
    video_tensor = torch.from_numpy(video).float() / 127.5 - 1.0
    video_tensor = rearrange(video_tensor, "f h w c -> 1 c f h w")
    
    # Masked video (black out mask==1 region)
    masked_video = video.copy()
    for i in range(num_frames):
        masked_video[i][mask > 0.5] = 0
    masked_video_tensor = torch.from_numpy(masked_video).float() / 127.5 - 1.0
    masked_video_tensor = rearrange(masked_video_tensor, "f h w c -> 1 c f h w")
    
    # Mask: [H, W] -> [1, 1, H, W]
    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    
    # Reference: [H, W, C] -> [1, C, 1, H, W]
    ref_tensor = torch.from_numpy(ref_processed).float() / 127.5 - 1.0
    ref_tensor = rearrange(ref_tensor, "h w c -> 1 c 1 h w")
    
    print("=" * 60)
    print("Encoding to latent space...")
    print("=" * 60)
    
    # Debug: save masked video frame to verify masking
    debug_dir = os.path.dirname(output_path)
    debug_frame = ((masked_video_tensor[0, :, 0].permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
    debug_frame_bgr = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(debug_dir, "debug_masked_frame.png"), debug_frame_bgr)
    print(f"[DEBUG] Saved masked frame to {os.path.join(debug_dir, 'debug_masked_frame.png')}")
    
    # Debug: save reference processed
    debug_ref = ref_processed.copy()
    cv2.imwrite(os.path.join(debug_dir, "debug_ref_processed.png"), cv2.cvtColor(debug_ref, cv2.COLOR_RGB2BGR))
    print(f"[DEBUG] Saved ref processed to {os.path.join(debug_dir, 'debug_ref_processed.png')}")
    
    with torch.no_grad():
        # Encode masked video
        masked_video_latents = prepare_latents(vae, masked_video_tensor, device, load_dtype, generator)
        
        # Encode reference
        ref_latents = prepare_latents(vae, ref_tensor, device, load_dtype, generator)
        
        # Prepare mask latents
        mask_latents = prepare_mask_latents(mask_tensor.to(device), num_frames)
        mask_latents = mask_latents.to(device=device, dtype=load_dtype)
    
    print(f"Masked video latents: {masked_video_latents.shape}")
    print(f"Reference latents: {ref_latents.shape}")
    print(f"Mask latents: {mask_latents.shape}")
    
    print("=" * 60)
    print("Running inference...")
    print("=" * 60)
    
    # Run inference
    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=H,
        width=W,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        masked_video_latents=masked_video_latents,
        mask_latents=mask_latents,
        ref_latents=ref_latents,
        output_type="pt",
        return_dict=True,
    )
    
    # Get output video and convert to uint8
    output_video = output.frames[0]  # [F, C, H, W] in [0, 1] float
    output_video = (
        (output_video * 255.0)
        .to(torch.uint8)
        .movedim(1, -1)  # [F, C, H, W] -> [F, H, W, C]
        .detach()
        .cpu()
        .numpy()
    )
    
    # Resize output back to original size
    output_video = np.array([cv2.resize(frame, (orig_W, orig_H)) for frame in output_video])
    
    # Save output
    save_video(output_video, output_path, fps)
    
    print("=" * 60)
    print("Done!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="S2V-Style Video Inpainting Inference")
    
    parser.add_argument("--video_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/2/input.mp4", help="Input video path")
    parser.add_argument("--mask_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/2/maskVid_frame0.png", help="Mask image path (single image)")
    parser.add_argument("--ref_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/2/ref.jpg", help="Reference image path")
    parser.add_argument("--output_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/2/rope_2000steps.mp4", help="Output video path")
    parser.add_argument("--prompt", type=str, default="A flower is gently swaying in nature.", help="Prompt for generation")
 
    # parser.add_argument("--video_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0016/original.mp4", help="Input video path")
    # parser.add_argument("--mask_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0016/mask.png", help="Mask image path (single image)")
    # parser.add_argument("--ref_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0016/ref_image_aug.png", help="Reference image path")
    # parser.add_argument("--output_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0016/rope_1000steps_resize_cfg3.0.mp4", help="Output video path")
    # parser.add_argument("--prompt", type=str, default="A man is talking.", help="Prompt for generation")
    
    parser.add_argument("--base_model_path", type=str, 
                        default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/ckps/wanErase/base",
                        help="Path to base Wan model")
    parser.add_argument("--transformer_path", type=str,
                        default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/ckps/wanErase/checkpoint-step00105000",
                        help="Path to transformer checkpoint")
    parser.add_argument("--lora_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/outputs/inpaint_s2v_style/inpaint_s2v_v1/checkpoint-step00002000", help="Path to LoRA checkpoint")

    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--infer_h", type=int, default=480, help="Inference height")
    parser.add_argument("--infer_w", type=int, default=640, help="Inference width")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    infer(
        video_path=args.video_path,
        mask_path=args.mask_path,
        ref_path=args.ref_path,
        output_path=args.output_path,
        prompt=args.prompt,
        base_model_path=args.base_model_path,
        transformer_path=args.transformer_path,
        lora_path=args.lora_path,
        negative_prompt=args.negative_prompt,
        infer_h=args.infer_h,
        infer_w=args.infer_w,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
