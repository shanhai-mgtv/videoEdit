import os
import argparse
from typing import Optional

import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from transformers import UMT5EncoderModel, AutoTokenizer
from moviepy import ImageSequenceClip

from models.autoencoder_kl_wan import AutoencoderKLWan
from models.transformer_wan import WanTransformer3DModel
from models.flow_match import FlowMatchScheduler
from pipelines.pipeline_wan_inpainting import (
    WanPipeline,
    retrieve_latents,
    normalize_latents,
    denormalize_latents,
)
from peft import PeftModel


def prepare_latents(
    vae: AutoencoderKLWan,
    image_or_video: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
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

    latents = retrieve_latents(
        latents,
        generator,
        sample_mode="sample",  # 与训练一致
    )

    latents = normalize_latents(
        latents=latents,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )

    return latents.to(dtype=dtype, device=device)


def post_latents(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    device = device or vae.device
    dtype = dtype or vae.dtype

    latents = latents.to(device=device, dtype=vae.dtype)
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    )
    latents_std = (
        torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device, vae.dtype)
    )

    latents = denormalize_latents(
        latents=latents,
        latents_mean=latents_mean,
        latents_std=latents_std,
    )

    latents = latents.to(dtype=dtype, device=device)
    video = vae.decode(latents, return_dict=False)[0]
    return video.to(dtype=dtype, device=device)


def save_video_with_numpy(video, path, fps):
    frames = []
    for img in video:
        frames.append(img)
    clip = ImageSequenceClip(frames, fps=fps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clip.write_videofile(path, codec="libx264", bitrate="10M")


def read_video_cv2(video_path):
    """读取视频，返回 RGB 帧数组和 fps"""
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
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Error: Could not read mask image {mask_path}")
    mask = (mask > 5).astype(np.uint8) * 255
    return mask


def read_ref_image(ref_path):
    ref = cv2.imread(ref_path)
    if ref is None:
        raise ValueError(f"Error: Could not read ref image {ref_path}")
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    return ref_rgb


def expand_mask_to_video(mask_image: np.ndarray, num_frames: int) -> np.ndarray:
    mask_with_channel = mask_image[np.newaxis, :, :]
    mask_video = np.repeat(mask_with_channel[np.newaxis, :, :, :], num_frames, axis=0)
    mask_video = mask_video.squeeze(1)  # (F, 1, H, W)
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

    load_dtype = torch.bfloat16

    print("=" * 50)
    print("Loading models...")
    print("=" * 50)

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        model_path,
        subfolder="text_encoder",
        torch_dtype=load_dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_path,
        subfolder="vae",
        torch_dtype=load_dtype,
    )
    
    transformer = WanTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    
    if lora_path is not None and os.path.exists(lora_path):
        print(f"Loading LoRA weights from {lora_path}")
        transformer = PeftModel.from_pretrained(transformer, lora_path)
    
    scheduler = FlowMatchScheduler(shift=7, sigma_min=0.0, extra_one_step=True)
    pipeline = WanPipeline(
        tokenizer=tokenizer,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)

    print("=" * 50)
    print("Loading input data...")
    print("=" * 50)

    # 读取输入数据
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

    # 获取原始视频尺寸
    H, W, _ = orig_video[0].shape

    if mask_image.shape[0] != H or mask_image.shape[1] != W:
        mask_image = cv2.resize(mask_image, (W, H), interpolation=cv2.INTER_NEAREST)
        print(f"Resized mask to video size: {mask_image.shape}")
    
    # 调整 ref_image 到视频尺寸
    if ref_image.shape[0] != H or ref_image.shape[1] != W:
        ref_image = cv2.resize(ref_image, (W, H), interpolation=cv2.INTER_LINEAR)
        print(f"Resized ref image to video size: {ref_image.shape}")
    
    # 调整 ref_mask 到视频尺寸
    if ref_mask is not None and (ref_mask.shape[0] != H or ref_mask.shape[1] != W):
        ref_mask = cv2.resize(ref_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        print(f"Resized ref mask to video size: {ref_mask.shape}")

    video_frame_len = len(orig_video) // infer_len * infer_len
    if video_frame_len == 0:
        video_frame_len = len(orig_video)
        print(f"Warning: Video has fewer frames than infer_len ({infer_len}), using all {video_frame_len} frames")

    orig_mask = expand_mask_to_video(mask_image, len(orig_video))
    print(f"Expanded mask video shape: {orig_mask.shape}")

    print("=" * 50)
    print("Starting inference...")
    print("=" * 50)

    generated_frames = []
    for start_idx in range(0, video_frame_len, infer_len):
        end_idx = min(start_idx + infer_len, len(orig_video))
        actual_len = end_idx - start_idx

        if actual_len < infer_len:
            print(f"Skipping incomplete segment: frames {start_idx}-{end_idx} (only {actual_len} frames)")
            continue

        print(f"Processing frames {start_idx} to {end_idx}...")

        batch_orig_frames = orig_video[start_idx:end_idx]  # [F, H, W, C]
        batch_orig_mask = orig_mask[start_idx:end_idx]  # [F, 1, H, W]

        # 处理 mask
        mask_seq = torch.from_numpy(batch_orig_mask).to(device)
        mask_seq = F.interpolate(
            mask_seq.float(), size=(infer_h, infer_w), mode="nearest-exact"
        )
        mask_seq = mask_seq.to(torch.float16) / 255.0

        # 处理 mask 到 latent 尺寸
        first_frame_mask = mask_seq[0:1, :, :]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=0, repeats=4)
        mask_lat_size = torch.concat([first_frame_mask, mask_seq[1:, :, :, :]], dim=0)
        mask_lat_size = F.interpolate(
            mask_lat_size, scale_factor=1 / 16, mode="nearest-exact"
        )

        num_frames, _, latent_height, latent_width = mask_lat_size.shape
        mask_lat_size = mask_lat_size.view(
            1, num_frames // 4, 4, latent_height, latent_width
        ) 
        mask_lat_size = mask_lat_size.transpose(1, 2)  # [B, C, F//4, H, W]

        with torch.no_grad():
            with torch.autocast("cuda", dtype=load_dtype):
                video_transforms = transforms.Compose(
                    [
                        transforms.Lambda(lambda x: x / 255.0),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ]
                )

                cond_video = torch.from_numpy(
                    np.stack(batch_orig_frames, axis=0)
                ).permute(0, 3, 1, 2)  # [F, C, H, W]

                cond_video = F.interpolate(
                    cond_video.float(), size=(infer_h, infer_w), mode="bicubic"
                )
                cond_video = torch.stack(
                    [video_transforms(x) for x in cond_video], dim=0
                )

                with torch.inference_mode():
                    image_or_video = cond_video.to(device=device, dtype=load_dtype)
                    # masked_video: mask==1 区域置零
                    masked_video = image_or_video * (1 - mask_seq)

                    # ========== DEBUG: 检查输入数据范围 ==========
                    print(f"[DEBUG] image_or_video: shape={image_or_video.shape}, min={image_or_video.min():.4f}, max={image_or_video.max():.4f}")
                    print(f"[DEBUG] mask_seq: shape={mask_seq.shape}, min={mask_seq.min():.4f}, max={mask_seq.max():.4f}, sum={mask_seq.sum():.0f}")
                    print(f"[DEBUG] masked_video: shape={masked_video.shape}, min={masked_video.min():.4f}, max={masked_video.max():.4f}")

                    # masked_video: [F, C, H, W] -> [1, C, F, H, W]
                    masked_video_5d = masked_video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]

                    # 编码 masked_video 到 latent
                    masked_video_latents = prepare_latents(vae, masked_video_5d)
                    masked_video_latents = masked_video_latents.to(dtype=load_dtype)
                    print(f"[DEBUG] masked_video_latents: shape={masked_video_latents.shape}, min={masked_video_latents.min():.4f}, max={masked_video_latents.max():.4f}")
                    
                    # 准备 ref_image latent
                    # ref_image: [H, W, C] -> [1, C, H, W] -> resize -> [1, C, infer_h, infer_w]
                    ref_tensor = torch.from_numpy(ref_image).permute(2, 0, 1).unsqueeze(0).float()
                    ref_tensor = F.interpolate(ref_tensor, size=(infer_h, infer_w), mode="bicubic")
                    ref_tensor = ref_tensor / 255.0 * 2 - 1
                    ref_tensor = ref_tensor.to(device=device, dtype=load_dtype)
                    print(f"[DEBUG] ref_tensor: shape={ref_tensor.shape}, min={ref_tensor.min():.4f}, max={ref_tensor.max():.4f}")
                    
                    # ref_img: [B, C, H, W] -> [B, C, 1, H, W] for VAE encoding
                    ref_img_video = ref_tensor.unsqueeze(2)  # [1, C, 1, H, W]
                    ref_latents = prepare_latents(vae, ref_img_video)
                    ref_latents = ref_latents.to(dtype=load_dtype)  # [1, 16, 1, h, w]
                    print(f"[DEBUG] ref_latents: shape={ref_latents.shape}, min={ref_latents.min():.4f}, max={ref_latents.max():.4f}")
                    
                    # 如果提供了 ref_mask，使用它；否则使用全 1（使用完整参考图）
                    if ref_mask is not None:
                        ref_mask_tensor = torch.from_numpy(ref_mask).float().unsqueeze(0).unsqueeze(0)
                        ref_mask_tensor = F.interpolate(ref_mask_tensor, size=(infer_h, infer_w), mode="nearest-exact")
                        ref_mask_tensor = ref_mask_tensor / 255.0
                    else:
                        ref_mask_tensor = torch.ones(1, 1, infer_h, infer_w)
                    ref_mask_tensor = ref_mask_tensor.to(device=device, dtype=load_dtype)
                    print(f"[DEBUG] ref_mask_tensor: shape={ref_mask_tensor.shape}, min={ref_mask_tensor.min():.4f}, max={ref_mask_tensor.max():.4f}")
                    
                    # 缩放到 latent 尺寸
                    mask_img_lat_size = F.interpolate(ref_mask_tensor, scale_factor=1/16, mode="nearest-exact")
                    mask_img_lat_size = mask_img_lat_size.unsqueeze(2).repeat(1, 4, 1, 1, 1)  # [1, 4, 1, h, w]
                    print(f"[DEBUG] mask_img_lat_size: shape={mask_img_lat_size.shape}, min={mask_img_lat_size.min():.4f}, max={mask_img_lat_size.max():.4f}")
                    print(f"[DEBUG] mask_lat_size (cond_masks): shape={mask_lat_size.shape}, min={mask_lat_size.min():.4f}, max={mask_lat_size.max():.4f}")
                    print("=" * 60)

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
                    ref_latents=ref_latents,
                    mask_img_latents=mask_img_lat_size,
                    output_type="latent",
                    strength=strength,
                ).frames

                # ========== DEBUG: 检查 pipeline 输出 ==========
                print(f"[DEBUG] gen_latent: shape={gen_latent.shape}, min={gen_latent.min():.4f}, max={gen_latent.max():.4f}, mean={gen_latent.mean():.4f}, std={gen_latent.std():.4f}")
                
                with torch.inference_mode():
                    gen_video = post_latents(vae, gen_latent)
                    print(f"[DEBUG] gen_video (after post_latents): shape={gen_video.shape}, min={gen_video.min():.4f}, max={gen_video.max():.4f}, mean={gen_video.mean():.4f}, std={gen_video.std():.4f}")
                    
                    gen_video = pipeline.video_processor.postprocess_video(
                        gen_video, output_type="pt"
                    )[0]
                    print(f"[DEBUG] gen_video (after postprocess): shape={gen_video.shape}, min={gen_video.min():.4f}, max={gen_video.max():.4f}, mean={gen_video.mean():.4f}, std={gen_video.std():.4f})")

                gen_video = (
                    (gen_video * 255.0)
                    .to(torch.uint8)
                    .movedim(1, -1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                print(f"[DEBUG] gen_video (final numpy): shape={gen_video.shape}, min={gen_video.min()}, max={gen_video.max()}, mean={gen_video.mean():.2f}, std={gen_video.std():.2f}")

                generated_frames += [
                    cv2.resize(video_frame, (W, H)) for video_frame in gen_video
                ]

    print("=" * 50)
    print(f"Saving output video to {output_path}...")
    print("=" * 50)

    save_video_with_numpy(generated_frames, output_path, fps)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Video inpainting with mask image and reference image")
    # parser.add_argument("--video_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/inpaint_lora_1222_change_the_refmask/inpaint_lora_v1/visualizations/vis_step_00000004/target_video.mp4", help="Input video path")
    # parser.add_argument("--mask_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/inpaint_lora_1222_change_the_refmask/inpaint_lora_v1/visualizations/vis_step_00000004/mask_image.png", help="Mask image path (single image)")
    # parser.add_argument("--ref_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/inpaint_lora_1222_change_the_refmask/inpaint_lora_v1/visualizations/vis_step_00000004/ref_image.png", help="Reference image path")
    # parser.add_argument("--output_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/inpaint_lora_1222_change_the_refmask/inpaint_lora_v1/visualizations/vis_step_00000004/out_2.mp4", help="Output video path")
    # parser.add_argument("--ref_mask_path",type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/inpaint_lora_1222_change_the_refmask/inpaint_lora_v1/visualizations/vis_step_00000004/ref_masked_image.png", )
    parser.add_argument("--video_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0012/original.mp4", help="Input video path")
    parser.add_argument("--mask_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0012/mask.png", help="Mask image path (single image)")
    parser.add_argument("--ref_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0012/ref_image_aug.png", help="Reference image path")
    parser.add_argument("--output_path", type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0012/out_0002_argmax_refnorm_again.mp4", help="Output video path")
    parser.add_argument("--ref_mask_path",type=str, default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/validation_demo/sample_0012/ref_mask_aug.png", )
    parser.add_argument("--prompt", type=str, default="A man is talking.", help="Prompt for generation")
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
        # default = None,
        default="/mnt/shanhai-ai/shanhai-workspace/lihaoran/project/code/videoEdit/videoEdit/wan_eraser/outputs/inpaint_lora_1222_change_the_refmask/inpaint_lora_v1/checkpoint-step00002800/lora_adapter",
        help="LoRA adapter path (directory containing adapter_model.safetensors and adapter_config.json)",
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
