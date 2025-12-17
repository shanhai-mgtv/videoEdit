import os
from typing import Any, Dict, List, Optional, Tuple, Union

import diffusers
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers.optimization import get_scheduler
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
        sample_mode="argmax",
    )
    # latents = latents.latent_dist.mode()

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


def read_video_cv2(video_path, mask=False):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return np.array([])

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        # Read a frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            break

        # Convert BGR to RGB
        if mask:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = (frame_gray > 5).astype(np.uint8) * 255
            frame_gray = frame_gray[None, :, :]
            frames.append(frame_gray)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    # Release the video capture object
    cap.release()

    return np.array(frames), fps


def infer():
    load_dtype = torch.bfloat16
    device = "cuda"
    model_path = "/mnt/cfs/shanhai/liuh/Wan2.2-TI2V-5B-Diffusers/"
    transformer_path = (
        "/mnt/cfs/shanhai/wangsiyuan/wan_2/wan_eraser/outputs/gogogo_argmax/checkpoint-step00005000"
    )

    #model_path = "/home/yutian6/workspace/userdata/model_weights/wan_earse/erase_25_08_05_wan2.1_00004500"
    #transformer_path = "/home/yutian6/workspace/userdata/model_weights/wan_earse/erase_25_08_05_wan2.1_00004500"

    # init model
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
    scheduler = FlowMatchScheduler(shift=7, sigma_min=0.0, extra_one_step=True)
    pipeline = WanPipeline(
        tokenizer=tokenizer,
        vae=vae,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(42)

    # preprocess data
    video_path = "/mnt/cfs/shanhai/wangsiyuan/wan_eraser/data/10041495.mp4"
    mask_path = "/mnt/cfs/shanhai/wangsiyuan/wan_eraser/data/10041495_mask.mp4"

    #video_path = "datas/demo/640p/long1.mp4"
    #mask_path = "datas/demo/640p/long1_sam2.mp4"

    prompt = "A monkey standing on the guardrail"
    negative_prompt = "Colorful color tone, overexposure, static, blurry details, subtitles, style, artwork, picture, static, overall graying, worst quality, low-quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, finger fusion, still image, cluttered background, three legs, many people in the background, walking backwards, no noise"
    orig_video, fps = read_video_cv2(video_path)
    orig_mask, _ = read_video_cv2(mask_path, mask=True)
    infer_len = 81
    H, W, _ = orig_video[0].shape

    # infer_h = 720
    # infer_w = 1280

    infer_h = 480
    infer_w = 640
    video_frame_len = len(orig_video) // infer_len * infer_len
    generated_frames = []
    for start_idx in range(0, video_frame_len, infer_len):
        end_idx = start_idx + infer_len
        print("start_idx and end_idx: ", start_idx, end_idx)
        batch_orig_frames = orig_video[list(range(start_idx, end_idx))]  # [f h w c]
        batch_orig_mask = orig_mask[list(range(start_idx, end_idx))]  # [f c h w ]
        # mask_seq = rearrange(batch_orig_mask,"f c h w -> f c h w")
        mask_seq = torch.from_numpy(batch_orig_mask).to(device)
        mask_seq = F.interpolate(
            mask_seq, size=(infer_h, infer_w), mode="nearest-exact"
        )
        mask_seq = mask_seq.to(torch.float16) / 255.0
        print(mask_seq.shape)
        first_frame_mask = mask_seq[0:1, :, :]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=0, repeats=4)
        mask_lat_size = torch.concat(
            [first_frame_mask, mask_seq[1:, :, :, :]], dim=0
        )  # f c h w
        mask_lat_size = F.interpolate(
            mask_lat_size, scale_factor=1 / 16, mode="nearest-exact"
        )
        print(mask_lat_size.shape)
        num_frames, _, latent_height, latent_width = mask_lat_size.shape
        mask_lat_size = mask_lat_size.view(
            1, num_frames // 4, 4, latent_height, latent_width
        )  # b f c h w
        mask_lat_size = mask_lat_size.transpose(1, 2)  # b c f h w
        with torch.no_grad():
            with torch.autocast("cuda", dtype=load_dtype):
                video_transforms = transforms.Compose(
                    [
                        transforms.Lambda(lambda x: x / 255.0),
                        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ]
                )
                cond_video = torch.from_numpy(
                    np.stack(batch_orig_frames, axis=0)
                ).permute(
                    0, 3, 1, 2
                )  # F C H W
                cond_video = F.interpolate(
                    cond_video, size=(infer_h, infer_w), mode="bicubic"
                )
                cond_video = torch.stack(
                    [video_transforms(x) for x in cond_video], dim=0
                )
                with torch.inference_mode():
                    # image_or_video = cond_video.to(device=device, dtype=load_dtype).unsqueeze(0) #[B, F, C, H, W]
                    image_or_video = cond_video.to(device=device, dtype=load_dtype)
                    image_or_video = image_or_video * (1 - mask_seq)
                    # image_or_video = image_or_video.unsqueeze(0).permute(
                    #     0, 2, 1, 3, 4
                    # ).contiguous()  # [B, F, C, H, W] -> [B, C, F, H, W]
                    image_or_video = pipeline.video_processor.preprocess_video(
                        image_or_video,
                        height=image_or_video.size(2),
                        width=image_or_video.size(3),
                    )  # (B,C,F,H,W)

                    cond_latents = prepare_latents(vae, image_or_video)
                    cond_latents = cond_latents.to(dtype=load_dtype)
                gen_latent = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=infer_h,
                    width=infer_w,
                    num_frames=81,
                    num_inference_steps=40,
                    guidance_scale=3.0,  # 3.0
                    generator=generator,
                    cond_latents=cond_latents,
                    cond_masks=mask_lat_size,
                    output_type="latent",
                    strength=0.8
                ).frames

                with torch.inference_mode():
                    gen_video = post_latents(vae, gen_latent)
                    gen_video = pipeline.video_processor.postprocess_video(
                        gen_video, output_type="pt"
                    )[0]
                gen_video = (
                    (gen_video * 255.0)
                    .to(torch.uint8)
                    .movedim(1, -1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                generated_frames += [
                    cv2.resize(video_frame, (W, H)) for video_frame in gen_video
                ]
    # output_path = "data/result.mp4"
    output_path = "datas/demo/640p/long1_gen.mp4"
    save_video_with_numpy(generated_frames, output_path, fps)


if __name__ == "__main__":
    infer()
