"""
Video Inpainting Inference Script with FIXED Reference Frame RoPE Position at 60

Key difference from other inference scripts:
- Reference frame RoPE position is FIXED at index 60 instead of sequential after masked video
- Uses WanTransformer3DModelFixedRef and WanPipelineFixedRef

Inputs:
    1. Original video (gt)
    2. Mask image (binary mask)
    3. Reference image (foreground cropped by bbox)
    4. Optional: Reference mask
"""

import os
import sys
import argparse
from typing import Optional

import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from transformers import UMT5EncoderModel, AutoTokenizer
from moviepy import ImageSequenceClip
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.autoencoder_kl_wan import AutoencoderKLWan
from models.transformer_wan_rope import WanTransformer3DModelFixedRef
from models.flow_match import FlowMatchScheduler

from pipelines.pipeline_wan_inpainting_fixed_ref import (
    WanPipelineFixedRef,
    retrieve_latents,
    normalize_latents,
    denormalize_latents,
)

def prepare_latents(
    vae: AutoencoderKLWan,
    image_or_video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode image or video to latent space."""
    device = device or vae.device
    dtype = dtype or vae.dtype

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    )

    latents = vae.encode(image_or_video)
    latents = retrieve_latents(latents, generator, sample_mode="sample")
    latents = normalize_latents(latents=latents, latents_mean=latents_mean, latents_std=latents_std)

    return latents.to(dtype=dtype, device=device)


def post_latents(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Decode latents to video."""
    device = device or vae.device
    dtype = dtype or vae.dtype

    latents = latents.to(device=device, dtype=vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    )

    latents = denormalize_latents(latents=latents, latents_mean=latents_mean, latents_std=latents_std)
    latents = latents.to(dtype=dtype, device=device)
    video = vae.decode(latents, return_dict=False)[0]
    return video.to(dtype=dtype, device=device)


def save_video_with_numpy(video, path, fps):
    """Save video frames to file."""
    frames = []
    for img in video:
        frames.append(img)
    clip = ImageSequenceClip(frames, fps=fps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clip.write_videofile(path, codec="libx264", bitrate="10M")


def read_video_cv2(video_path):
    """Read video file and return RGB frames and fps."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return np.array([]), 0

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


def read_mask_image(mask_path):
    """Read mask image and binarize."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Error: Could not read mask image {mask_path}")
    mask = (mask > 5).astype(np.uint8) * 255
    return mask


def read_ref_image(ref_path):
    """Read reference image."""
    ref = cv2.imread(ref_path)
    if ref is None:
        raise ValueError(f"Error: Could not read ref image {ref_path}")
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    return ref_rgb


def expand_mask_to_video(mask_image: np.ndarray, num_frames: int) -> np.ndarray:
    """Expand single mask image to video mask."""
    mask_with_channel = mask_image[np.newaxis, :, :]
    mask_video = np.repeat(mask_with_channel[np.newaxis, :, :, :], num_frames, axis=0)
    mask_video = mask_video.squeeze(1)
    if len(mask_video.shape) == 3:
        mask_video = mask_video[:, np.newaxis, :, :]
    return mask_video


def infer(
    video_path: str,
    mask_path: str,
    ref_path: str,
    output_path: str,
    model_path: str,
    transformer_path: str,
    lora_path: str = None,
    ref_mask_path: str = None,
    prompt: str = "",
    negative_prompt: str = "Colorful color tone, overexposure, static, blurry details, subtitles, style, artwork, picture, static, overall graying, worst quality, low-quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, finger fusion, still image, cluttered background, three legs, many people in the background, walking backwards, no noise",
    infer_h: int = 480,
    infer_w: int = 640,
    infer_len: int = 81,
    num_inference_steps: int = 40,
    guidance_scale: float = 3.0,
    strength: float = 0.8,
    seed: int = 42,
    device: str = "cuda",
):
    """
    Main inference function with FIXED reference frame RoPE position at 60.
    """
    load_dtype = torch.bfloat16

    print("=" * 60)
    print("Video Inpainting with FIXED Reference RoPE Position at 60")
    print("=" * 60)
    print("Loading models...")

    # Load models
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path, subfolder="text_encoder", torch_dtype=load_dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_path, subfolder="vae", torch_dtype=load_dtype
    )
    
    # Load transformer with fixed ref RoPE
    transformer = WanTransformer3DModelFixedRef.from_pretrained(
        transformer_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )

    # Handle LoRA weights if provided
    if lora_path is not None and os.path.exists(lora_path):
        # Find the safetensors file
        if os.path.isdir(lora_path):
            # Look for .safetensors file in the directory
            safetensor_files = [f for f in os.listdir(lora_path) if f.endswith('.safetensors')]
            if not safetensor_files:
                raise ValueError(f"No .safetensors file found in {lora_path}")
            lora_weights_path = os.path.join(lora_path, safetensor_files[0])
        else:
            lora_weights_path = lora_path
        
        print(f"Loading LoRA weights from {lora_weights_path}")
        
        lora_config = LoraConfig(
            r=128,
            lora_alpha=128,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )
        transformer = get_peft_model(transformer, lora_config)
        lora_state_dict = load_file(lora_weights_path)
        fixed_state_dict = {}
        for key, value in lora_state_dict.items():
            if key.startswith("transformer."):
                new_key = key[len("transformer."):]
                fixed_state_dict[new_key] = value
            else:
                fixed_state_dict[key] = value
        
        print(f"Fixed {len(fixed_state_dict)} keys (removed 'transformer.' prefix)")
        set_peft_model_state_dict(transformer, fixed_state_dict)
        print(f"LoRA weights loaded successfully")

    # Create pipeline with fixed ref RoPE
    scheduler = FlowMatchScheduler(shift=7, sigma_min=0.0, extra_one_step=True)
    pipeline = WanPipelineFixedRef(
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

    # Read input data
    orig_video, fps = read_video_cv2(video_path)
    mask_image = read_mask_image(mask_path)
    ref_image = read_ref_image(ref_path)
    
    if ref_mask_path is not None and os.path.exists(ref_mask_path):
        ref_mask = read_mask_image(ref_mask_path)
        print(f"Ref mask shape: {ref_mask.shape}")
    else:
        ref_mask = None
        print("No ref_mask provided, using full reference image")

    print(f"Video shape: {orig_video.shape}, FPS: {fps}")
    print(f"Mask shape: {mask_image.shape}")
    print(f"Ref image shape: {ref_image.shape}")

    # Get original video dimensions
    H, W, _ = orig_video[0].shape

    if mask_image.shape[0] != H or mask_image.shape[1] != W:
        mask_image = cv2.resize(mask_image, (W, H), interpolation=cv2.INTER_NEAREST)
        print(f"Resized mask to video size: {mask_image.shape}")
    
    if ref_image.shape[0] != H or ref_image.shape[1] != W:
        ref_image = cv2.resize(ref_image, (W, H), interpolation=cv2.INTER_LINEAR)
        print(f"Resized ref image to video size: {ref_image.shape}")
    
    if ref_mask is not None and (ref_mask.shape[0] != H or ref_mask.shape[1] != W):
        ref_mask = cv2.resize(ref_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        print(f"Resized ref mask to video size: {ref_mask.shape}")

    video_frame_len = len(orig_video) // infer_len * infer_len
    if video_frame_len == 0:
        video_frame_len = len(orig_video)
        print(f"Warning: Video has fewer frames than infer_len ({infer_len}), using all {video_frame_len} frames")

    orig_mask = expand_mask_to_video(mask_image, len(orig_video))
    print(f"Expanded mask video shape: {orig_mask.shape}")

    print("=" * 60)
    print("Starting inference with FIXED RoPE position at 60...")
    print("=" * 60)

    generated_frames = []
    for start_idx in range(0, video_frame_len, infer_len):
        end_idx = min(start_idx + infer_len, len(orig_video))
        actual_len = end_idx - start_idx

        if actual_len < infer_len:
            print(f"Skipping incomplete segment: frames {start_idx}-{end_idx} (only {actual_len} frames)")
            continue

        print(f"Processing frames {start_idx} to {end_idx}...")

        batch_orig_frames = orig_video[start_idx:end_idx]
        batch_orig_mask = orig_mask[start_idx:end_idx]

        # Process mask
        mask_seq = torch.from_numpy(batch_orig_mask).to(device)
        mask_seq = F.interpolate(mask_seq.float(), size=(infer_h, infer_w), mode="nearest-exact")
        mask_seq = mask_seq.to(torch.float16) / 255.0

        # Process mask to latent size
        first_frame_mask = mask_seq[0:1, :, :]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=0, repeats=4)
        mask_lat_size = torch.concat([first_frame_mask, mask_seq[1:, :, :, :]], dim=0)
        mask_lat_size = F.interpolate(mask_lat_size, scale_factor=1 / 16, mode="nearest-exact")

        num_frames, _, latent_height, latent_width = mask_lat_size.shape
        mask_lat_size = mask_lat_size.view(1, num_frames // 4, 4, latent_height, latent_width) 
        mask_lat_size = mask_lat_size.transpose(1, 2)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=load_dtype):
                video_transforms = transforms.Compose([
                    transforms.Lambda(lambda x: x / 255.0),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ])

                cond_video = torch.from_numpy(np.stack(batch_orig_frames, axis=0)).permute(0, 3, 1, 2)
                cond_video = F.interpolate(cond_video.float(), size=(infer_h, infer_w), mode="bicubic")
                cond_video = torch.stack([video_transforms(x) for x in cond_video], dim=0)

                with torch.inference_mode():
                    image_or_video = cond_video.to(device=device, dtype=load_dtype)
                    masked_video = image_or_video * (1 - mask_seq)

                    print(f"[DEBUG] image_or_video: shape={image_or_video.shape}, range=[{image_or_video.min():.4f}, {image_or_video.max():.4f}]")
                    print(f"[DEBUG] mask_seq: shape={mask_seq.shape}, range=[{mask_seq.min():.4f}, {mask_seq.max():.4f}]")

                    masked_video_5d = masked_video.permute(1, 0, 2, 3).unsqueeze(0)
                    masked_video_latents = prepare_latents(vae, masked_video_5d)
                    masked_video_latents = masked_video_latents.to(dtype=load_dtype)
                    
                    # Prepare ref_image latent
                    ref_tensor = torch.from_numpy(ref_image).permute(2, 0, 1).unsqueeze(0).float()
                    ref_tensor = F.interpolate(ref_tensor, size=(infer_h, infer_w), mode="bicubic")
                    ref_tensor = ref_tensor / 255.0 * 2 - 1
                    ref_tensor = ref_tensor.to(device=device, dtype=load_dtype)
                    
                    ref_img_video = ref_tensor.unsqueeze(2)
                    ref_latents = prepare_latents(vae, ref_img_video)
                    ref_latents = ref_latents.to(dtype=load_dtype)

                    # Prepare ref_mask
                    if ref_mask is not None:
                        ref_mask_tensor = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0)
                        ref_mask_tensor = F.interpolate(ref_mask_tensor, size=(infer_h, infer_w), mode="nearest-exact")
                        ref_mask_tensor = ref_mask_tensor / 255.0
                    else:
                        ref_mask_tensor = torch.ones(1, 1, infer_h, infer_w)
                    ref_mask_tensor = ref_mask_tensor.to(device=device, dtype=load_dtype)
                    
                    mask_img_lat_size = F.interpolate(ref_mask_tensor, scale_factor=1/16, mode="nearest-exact")
                    mask_img_lat_size = mask_img_lat_size.unsqueeze(2).repeat(1, 4, 1, 1, 1)

                    print(f"[DEBUG] masked_video_latents: shape={masked_video_latents.shape}")
                    print(f"[DEBUG] ref_latents: shape={ref_latents.shape}")
                    print(f"[DEBUG] mask_lat_size: shape={mask_lat_size.shape}")
                    print(f"[DEBUG] mask_img_lat_size: shape={mask_img_lat_size.shape}")

                gen_latent = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=infer_h,
                    width=infer_w,
                    num_frames=infer_len,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    cond_latents=masked_video_latents,
                    cond_masks=mask_lat_size,
                    mask_img_latents=mask_img_lat_size,
                    ref_latents=ref_latents,
                    output_type="latent",
                ).frames

                print(f"[DEBUG] gen_latent: shape={gen_latent.shape}, range=[{gen_latent.min():.4f}, {gen_latent.max():.4f}]")
                
                with torch.inference_mode():
                    gen_video = post_latents(vae, gen_latent)
                    gen_video = pipeline.video_processor.postprocess_video(gen_video, output_type="pt")[0]

                gen_video = (
                    (gen_video * 255.0)
                    .to(torch.uint8)
                    .movedim(1, -1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                generated_frames += [cv2.resize(video_frame, (W, H)) for video_frame in gen_video]

    print("=" * 60)
    print(f"Saving output video to {output_path}...")
    print("=" * 60)

    save_video_with_numpy(generated_frames, output_path, fps)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Video inpainting with FIXED reference frame RoPE position at 60")
    
    # Batch mode: process all demos in validation_demo directory
    parser.add_argument("--batch_mode", action="store_true", help="Process all demos in validation_demo directory")
    parser.add_argument("--demo_dir", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo", help="Directory containing demo folders")
    parser.add_argument("--output_suffix", type=str, default="rope_1800steps", help="Suffix for output video filename in batch mode")
    
    # Single mode arguments
    parser.add_argument("--video_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0004/original.mp4", help="Input video path")
    parser.add_argument("--mask_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0004/mask.png", help="Mask image path (single image)")
    parser.add_argument("--ref_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0004/ref_image_aug.png", help="Reference image path")
    parser.add_argument("--output_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0004/rope_600steps.mp4", help="Output video path")
    parser.add_argument("--ref_mask_path",type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0004/ref_mask_aug.png", )
    parser.add_argument("--prompt", type=str, default="A woman is talking.", help="Prompt for generation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/ckps/wanErase/base",
        help="Base model path",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/ckps/wanErase/checkpoint-step00105000",
        help="Transformer weights path",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/lora_rope_fixed60/inpaint_lora_rope/checkpoint-step00001800",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Colorful color tone, overexposure, static, blurry details, subtitles, style, artwork, picture, static, overall graying, worst quality, low-quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, finger fusion, still image, cluttered background, three legs, many people in the background, walking backwards, no noise",
        help="Negative prompt",
    )
    parser.add_argument("--infer_h", type=int, default=480, help="Inference height")
    parser.add_argument("--infer_w", type=int, default=640, help="Inference width")
    parser.add_argument("--infer_len", type=int, default=81, help="Frames per inference")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale")
    parser.add_argument("--strength", type=float, default=0.8, help="Strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    if args.batch_mode:
        # Batch mode: process all sample_* directories in demo_dir
        demo_dirs = sorted([
            d for d in os.listdir(args.demo_dir) 
            if os.path.isdir(os.path.join(args.demo_dir, d)) and d.startswith("sample_")
        ])
        
        print("=" * 60)
        print(f"BATCH MODE: Found {len(demo_dirs)} demos to process")
        print("=" * 60)
        
        for i, demo_name in enumerate(demo_dirs):
            demo_path = os.path.join(args.demo_dir, demo_name)
            
            video_path = os.path.join(demo_path, "original.mp4")
            mask_path = os.path.join(demo_path, "mask.png")
            ref_path = os.path.join(demo_path, "ref_image_aug.png")
            ref_mask_path = os.path.join(demo_path, "ref_mask_aug.png")
            output_path = os.path.join(demo_path, f"{args.output_suffix}.mp4")
            
            caption_path = os.path.join(demo_path, "caption.txt")
            if os.path.exists(caption_path):
                with open(caption_path, "r") as f:
                    prompt = f.read().strip()
                if not prompt:
                    prompt = args.prompt
            else:
                prompt = args.prompt
            
            if not os.path.exists(video_path):
                print(f"[{i+1}/{len(demo_dirs)}] Skipping {demo_name}: original.mp4 not found")
                continue
            if not os.path.exists(mask_path):
                print(f"[{i+1}/{len(demo_dirs)}] Skipping {demo_name}: mask.png not found")
                continue
            if not os.path.exists(ref_path):
                # Try ref_image.png as fallback
                ref_path_fallback = os.path.join(demo_path, "ref_image.png")
                if os.path.exists(ref_path_fallback):
                    ref_path = ref_path_fallback
                else:
                    print(f"[{i+1}/{len(demo_dirs)}] Skipping {demo_name}: ref_image not found")
                    continue
            
            # ref_mask is optional
            if not os.path.exists(ref_mask_path):
                ref_mask_path = None
            
            print("\n" + "=" * 60)
            print(f"[{i+1}/{len(demo_dirs)}] Processing: {demo_name}")
            print(f"  Video: {video_path}")
            print(f"  Output: {output_path}")
            print("=" * 60)
            
            try:
                infer(
                    video_path=video_path,
                    mask_path=mask_path,
                    ref_path=ref_path,
                    output_path=output_path,
                    model_path=args.model_path,
                    transformer_path=args.transformer_path,
                    lora_path=args.lora_path,
                    ref_mask_path=ref_mask_path,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    infer_h=args.infer_h,
                    infer_w=args.infer_w,
                    infer_len=args.infer_len,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    strength=args.strength,
                    seed=args.seed,
                    device=args.device,
                )
            except Exception as e:
                print(f"[{i+1}/{len(demo_dirs)}] Error processing {demo_name}: {e}")
                continue
        
        print("\n" + "=" * 60)
        print("BATCH MODE COMPLETED")
        print("=" * 60)
    else:
        # Single mode
        infer(
            video_path=args.video_path,
            mask_path=args.mask_path,
            ref_path=args.ref_path,
            output_path=args.output_path,
            model_path=args.model_path,
            transformer_path=args.transformer_path,
            lora_path=args.lora_path,
            ref_mask_path=args.ref_mask_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            infer_h=args.infer_h,
            infer_w=args.infer_w,
            infer_len=args.infer_len,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
