"""
Video Inpainting Training Script with FIXED Reference Frame RoPE Position at 60

Key difference from train_wan_sft_second.py:
- Reference frame RoPE position is FIXED at index 60 instead of sequential after masked video
- Uses WanTransformer3DModelFixedRef and WanPipelineFixedRef

Inputs:
    1. Original video (gt)
    2. Masked video (mask==1 region blacked out)
    3. Mask video (binary mask for all frames)
    4. Mask image (first frame mask)
    5. Reference image (first frame foreground cropped by bbox)

Three-branch temporal fusion:
    Branch 1: noisy_gt + masked_video + ref_img (temporal fusion)
    Branch 2: zeros_gt + mask_video + mask_img (temporal fusion)  
    Branch 3: zeros_gt + masked_video + ref_img (temporal fusion)

Final: Channel-wise concatenation of all branches -> Transformer

Based on train_wan_sft_second.py with fixed ref RoPE position at 60.
"""

import logging
import sys
import json
import warnings
import os
import random
import shutil
import typing
import math
import gc
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta

import torch
import cv2
import pyrallis
import transformers
import diffusers
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms.v2 as transforms
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from einops import rearrange, repeat
from PIL import Image
from decord import VideoReader
from tqdm.auto import tqdm
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from moviepy import ImageSequenceClip

from peft import LoraConfig, get_peft_model, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import re

from config import Config
from optim import get_optimizer, max_gradient
from models.transformer_wan_rope import WanTransformer3DModelFixedRef
from models.autoencoder_kl_wan import AutoencoderKLWan
from models.flow_match import FlowMatchScheduler
from pipelines.pipeline_wan_inpainting_fixed_ref import WanPipelineFixedRef, retrieve_latents, prompt_clean
from ema import EMAModel
# from utils_inference.video_writer import TensorSaveVideo


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    force=True,
    handlers=[logging.StreamHandler()],
)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)

# Norm layer prefixes for optional training
NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


# ============================================================================
# Utility Functions
# ============================================================================

def save_video_with_numpy(video, path, fps):
    frames = []
    for img in video:
        frames.append(img)
    clip = ImageSequenceClip(frames, fps=fps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clip.write_videofile(path, codec="libx264", bitrate="10M")


def read_video_cv2(video_path, mask=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return np.array([]), 0

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if mask:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = (frame_gray > 5).astype(np.uint8) * 255
            frame_gray = frame_gray[None, :, :]
            frames.append(frame_gray)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return np.array(frames), fps


def bytes_to_gigabytes(x: int) -> float:
    if x is not None:
        return x / 1024**3


def get_memory_statistics(precision: int = 3) -> typing.Dict[str, typing.Any]:
    memory_allocated = None
    memory_reserved = None
    max_memory_allocated = None
    max_memory_reserved = None

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)
        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)
    elif torch.backends.mps.is_available():
        memory_allocated = torch.mps.current_allocated_memory()
    else:
        logger.warning("No CUDA, MPS, or ROCm device found.")

    return {
        "memory_allocated": round(bytes_to_gigabytes(memory_allocated), ndigits=precision) if memory_allocated else None,
        "memory_reserved": round(bytes_to_gigabytes(memory_reserved), ndigits=precision) if memory_reserved else None,
        "max_memory_allocated": round(bytes_to_gigabytes(max_memory_allocated), ndigits=precision) if max_memory_allocated else None,
        "max_memory_reserved": round(bytes_to_gigabytes(max_memory_reserved), ndigits=precision) if max_memory_reserved else None,
    }


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
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, vae.dtype
    )
    latents = retrieve_latents(vae.encode(image_or_video), generator, sample_mode="sample")
    latents = (latents - latents_mean) * latents_std
    latents = latents.to(dtype=dtype)
    return latents


def get_nb_trainable_parameters(mod: torch.nn.Module):
    trainable_params = 0
    all_param = 0
    for _, param in mod.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


def get_t5_prompt_embeds(
    text_encoder: UMT5EncoderModel,
    tokenizer: AutoTokenizer,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt = [prompt_clean(u) for u in prompt]
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
    )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


# ============================================================================
# Three-Branch Fusion Functions
# ============================================================================

def create_ref_img_sequence(ref_img: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Expand reference image to a sequence by repeating along temporal dimension.
    
    Args:
        ref_img: [B, C, H, W] reference image
        num_frames: number of frames to expand to
        
    Returns:
        ref_seq: [B, C, F, H, W] reference sequence
    """
    # ref_img: [B, C, H, W] -> [B, C, 1, H, W] -> [B, C, F, H, W]
    return ref_img.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)


def create_mask_img_sequence(mask_img: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Expand mask image to a sequence by repeating along temporal dimension.
    
    Args:
        mask_img: [B, 1, H, W] mask image
        num_frames: number of frames to expand to
        
    Returns:
        mask_seq: [B, 1, F, H, W] mask sequence
    """
    return mask_img.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)


def temporal_fusion_branch1(
    noisy_latents: torch.Tensor,
    masked_video_latents: torch.Tensor,
    ref_latents: torch.Tensor,
) -> torch.Tensor:
    """
    Branch 1: noisy_gt + masked_video + ref_img (temporal fusion)
    
    Args:
        noisy_latents: [B, C, F, H, W] noisy ground truth latents
        masked_video_latents: [B, C, F, H, W] masked video latents
        ref_latents: [B, C, F, H, W] reference image latents (expanded to sequence)
        
    Returns:
        fused: [B, 3*C, F, H, W] temporally fused features
    """
    # Concatenate along channel dimension for temporal fusion
    return torch.cat([noisy_latents, masked_video_latents, ref_latents], dim=1)


def temporal_fusion_branch2(
    zeros_latents: torch.Tensor,
    mask_video: torch.Tensor,
    mask_img_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Branch 2: zeros_gt + mask_video + mask_img (temporal fusion)
    
    Args:
        zeros_latents: [B, C, F, H, W] zero-filled latents (same shape as gt)
        mask_video: [B, 1, F, H, W] mask video
        mask_img_seq: [B, 1, F, H, W] mask image expanded to sequence
        
    Returns:
        fused: [B, C+2, F, H, W] temporally fused features
    """
    return torch.cat([zeros_latents, mask_video, mask_img_seq], dim=1)


def temporal_fusion_branch3(
    zeros_latents: torch.Tensor,
    masked_video_latents: torch.Tensor,
    ref_latents: torch.Tensor,
) -> torch.Tensor:
    """
    Branch 3: zeros_gt + masked_video + ref_img (temporal fusion)
    
    Args:
        zeros_latents: [B, C, F, H, W] zero-filled latents
        masked_video_latents: [B, C, F, H, W] masked video latents
        ref_latents: [B, C, F, H, W] reference image latents (expanded to sequence)
        
    Returns:
        fused: [B, 3*C, F, H, W] temporally fused features
    """
    return torch.cat([zeros_latents, masked_video_latents, ref_latents], dim=1)


def channel_fusion(
    branch1_out: torch.Tensor,
    branch2_out: torch.Tensor,
    branch3_out: torch.Tensor,
) -> torch.Tensor:
    """
    Channel-wise fusion of all three branches.
    
    Args:
        branch1_out: [B, C1, F, H, W] output from branch 1
        branch2_out: [B, C2, F, H, W] output from branch 2
        branch3_out: [B, C3, F, H, W] output from branch 3
        
    Returns:
        fused: [B, C1+C2+C3, F, H, W] channel-fused features
    """
    return torch.cat([branch1_out, branch2_out, branch3_out], dim=1)


def prepare_mask_latent_size(mask_values: torch.Tensor, nframes: int) -> torch.Tensor:
    """
    Prepare mask for latent space size.
    
    Args:
        mask_values: [B*F, 1, H, W] mask values
        nframes: number of frames
        
    Returns:
        mask_video_latents: [B, 1, F//4, H//16, W//16] mask at latent size
    """
    # Downsample to latent size
    latent_masks = rearrange(
        F.interpolate(mask_values, scale_factor=1/16, mode="nearest-exact"),
        "(b f) c h w -> b c f h w", f=nframes
    )
    
    # Handle temporal compression (4x)
    first_frame_mask = latent_masks[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=4)
    mask_video_latents = torch.concat([first_frame_mask, latent_masks[:, :, 1:, :]], dim=2)
    
    batch_size, _, _, latent_height, latent_width = mask_video_latents.shape
    mask_video_latents = mask_video_latents.view(batch_size, -1, 4, latent_height, latent_width)
    mask_video_latents = mask_video_latents.transpose(1, 2)  # [B, C, F, H, W]
    
    return mask_video_latents


# ============================================================================
# Main Training Function
# ============================================================================

@pyrallis.wrap()
def main(cfg: Config):
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    output_dirpath = Path(cfg.experiment.output_dirpath) / cfg.experiment.run_id
    logging_dirpath = output_dirpath / "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=output_dirpath, logging_dir=logging_dirpath)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=cfg.ddp_kwargs.find_unused_parameters,
        gradient_as_bucket_view=cfg.ddp_kwargs.gradient_as_bucket_view,
        static_graph=cfg.ddp_kwargs.static_graph,
    )
    init_kwargs = InitProcessGroupKwargs(backend=cfg.ddp_kwargs.backend, timeout=timedelta(seconds=18000))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.hparams.gradient_accumulation_steps,
        mixed_precision=cfg.hparams.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )
    # tensor_writer = TensorSaveVideo()

    print(accelerator.state)
    accelerator.print("\nENVIRONMENT\n")
    accelerator.print(f"  Python .......................... {sys.version}")
    accelerator.print(f"  torch.__version__ ............... {torch.__version__}")
    accelerator.print(f"  torch.version.cuda .............. {torch.version.cuda}")
    accelerator.print(f"  torch.backends.cudnn.version() .. {torch.backends.cudnn.version()}\n")
    accelerator.print(f">> Run ID : {cfg.experiment.run_id!r}")

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.experiment.random_seed is not None:
        seed_with_rank = cfg.experiment.random_seed + accelerator.process_index
        set_seed(seed_with_rank)
        logger.info(f"Set seed {seed_with_rank} for process {accelerator.process_index}")

    if accelerator.num_processes > 1:
        logger.info("DDP VARS: ")
        logger.info(f"  WORLD_SIZE: {os.getenv('WORLD_SIZE', 'N/A')}")
        logger.info(f"  LOCAL_WORLD_SIZE: {os.getenv('LOCAL_WORLD_SIZE', 'N/A')}")
        logger.info(f"  RANK: {os.getenv('RANK', 'N/A')}")
        logger.info(f"  MASTER_ADDR: {os.getenv('MASTER_ADDR', 'N/A')}")
        logger.info(f"  MASTER_PORT: {os.getenv('MASTER_PORT', 'N/A')}")

    if accelerator.is_main_process:
        output_dirpath.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        logger.info(f"Saving config to {output_dirpath / 'config.yaml'}")
        yaml_cfg = pyrallis.dump(cfg)
        with open(output_dirpath / "config.yaml", "w") as f:
            f.write(yaml_cfg)

    logger.info(f"config = \n{pyrallis.dump(cfg)}")

    # ======================================================
    # Config validation
    # ======================================================
    def validate_config(cfg):
        """Validate critical config items before training."""
        errors = []
        
        # Data config validation
        if not hasattr(cfg.data, 'nframes') or cfg.data.nframes is None:
            errors.append("cfg.data.nframes is not set")
        elif cfg.data.nframes <= 0:
            errors.append(f"cfg.data.nframes must be positive, got {cfg.data.nframes}")
        
        if not hasattr(cfg.data, 'batch_size') or cfg.data.batch_size <= 0:
            errors.append(f"cfg.data.batch_size must be positive")
            
        # Model config validation
        if not hasattr(cfg.model, 'pretrained_model_name_or_path') or not cfg.model.pretrained_model_name_or_path:
            errors.append("cfg.model.pretrained_model_name_or_path is not set")
            
        if not hasattr(cfg.model, 'pretrained_model_transformer_name_or_path') or not cfg.model.pretrained_model_transformer_name_or_path:
            errors.append("cfg.model.pretrained_model_transformer_name_or_path is not set")
        
        # Training config validation
        if not hasattr(cfg.hparams, 'max_train_steps') or cfg.hparams.max_train_steps is None:
            if not hasattr(cfg.hparams, 'num_train_epochs') or cfg.hparams.num_train_epochs is None:
                errors.append("Either max_train_steps or num_train_epochs must be set")
        
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        logger.info("Config validation passed")
    
    validate_config(cfg)

    # ======================================================
    # 2. build model
    # ======================================================
    noise_scheduler = FlowMatchScheduler(shift=7, sigma_min=0.0, extra_one_step=True)
    noise_scheduler.set_timesteps(1000, training=True)

    load_dtype = torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=load_dtype,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=load_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    logger.info(f"Load transformer model from {cfg.model.pretrained_model_transformer_name_or_path!r}")

    # Load transformer with fixed ref RoPE (position 60 for reference frames)
    transformer = WanTransformer3DModelFixedRef.from_pretrained(
        cfg.model.pretrained_model_transformer_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    
    with torch.no_grad():
        logger.info("Verifying transformer input channels for three-branch temporal+channel fusion")
        initial_input_channels = transformer.config.in_channels
        logger.info(f"Transformer initial input channels: {initial_input_channels}")

    accelerator.wait_for_everyone()

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    logger.info(f"configured weight dtype: {weight_dtype!r}")

    if cfg.hparams.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ======================================================
    # LoRA Setup (if enabled)
    # ======================================================
    if cfg.experiment.use_lora:
        logger.info("Setting up LoRA fine-tuning...")
        transformer.requires_grad_(False)  # Freeze base model first
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=cfg.network.lora_rank,
            lora_alpha=cfg.network.lora_alpha,
            target_modules=cfg.network.target_modules or ["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=cfg.network.lora_dropout,
            init_lora_weights=cfg.network.init_lora_weights,
        )
        
        transformer = get_peft_model(transformer, lora_config)
        logger.info(f"LoRA config: rank={cfg.network.lora_rank}, alpha={cfg.network.lora_alpha}")
        transformer.print_trainable_parameters()
        
        # Load from existing LoRA checkpoint if specified
        if cfg.checkpointing.resume_from_lora_checkpoint:
            logger.info(f"Loading LoRA weights from {cfg.checkpointing.resume_from_lora_checkpoint}")
            transformer.load_adapter(cfg.checkpointing.resume_from_lora_checkpoint, adapter_name="default")
    else:
        logger.info("Full fine-tuning mode (no LoRA)")

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"dit dtype: {next(transformer.parameters()).dtype!r}")

    trainable_params, all_param = get_nb_trainable_parameters(transformer)
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    if cfg.hparams.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ema_model = None
    if accelerator.is_main_process and cfg.hparams.ema.use_ema:
        logger.info("Using EMA. Creating EMAModel.")
        ema_model_cls, ema_model_config = transformer.__class__, transformer.config
        ema_model = EMAModel(
            cfg.hparams.ema,
            accelerator,
            parameters=transformer_parameters,
            model_cls=ema_model_cls,
            model_config=ema_model_config,
            decay=cfg.hparams.ema.ema_decay,
            foreach=not cfg.hparams.ema.ema_foreach_disable,
        )
        logger.info(f"EMA model creation completed with {ema_model.parameter_count():,} parameters")

    accelerator.wait_for_everyone()

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        transformer_parameters,
        optimizer_name=cfg.hparams.optimizer_type,
        learning_rate=cfg.hparams.learning_rate,
        optimizer_args_str=cfg.hparams.optimizer_args,
        use_deepspeed=use_deepspeed_optimizer,
    )

    # ======================================================
    # 3. build dataset and dataloaders
    # ======================================================
    from dataloading.dataset_inpaint import InpaintingDataset, PreGeneratedInpaintingDataset
    
    # Enable debug saving for first few samples
    debug_save_dir = output_dirpath / "debug_data"

    # Choose dataset based on config
    use_pregenerated = getattr(cfg.data, 'use_pregenerated_data', False)
    if use_pregenerated:
        pregenerated_root = getattr(cfg.data, 'pregenerated_data_root', None)
        logger.info(f"Using PreGeneratedInpaintingDataset with pregenerated_data_root={pregenerated_root}")
        train_dataset = PreGeneratedInpaintingDataset(
            cfg.data,
            pregenerated_data_root=pregenerated_root,
            save_debug=True,
            debug_output_dir=str(debug_save_dir),
            debug_save_prob=0.001,
        )
    else:
        logger.info("Using InpaintingDataset with online mask generation")
        train_dataset = InpaintingDataset(
            cfg.data,
            save_debug=True,
            debug_output_dir=str(debug_save_dir),
            debug_save_prob=0.001,  # Save ~0.1% of samples for debugging
        )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.dataloader_kwargs.num_workers,
        pin_memory=True,
    )
    print("train_dataloader len after load: ", len(train_dataloader))

    # =======================================================
    # 4. distributed training preparation with accelerator
    # =======================================================
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return

        if cfg.hparams.ema.use_ema and ema_model is not None:
            primary_model = unwrap_model(transformer)
            ema_model_path = os.path.join(output_dir, "ema_model.pt")
            logger.info(f"Saving EMA model state to {ema_model_path!r}")
            try:
                ema_model.save_state_dict(ema_model_path)
            except Exception as e:
                logger.error(f"Error saving EMA model: {e!r}")

            logger.info("Saving EMA model to disk.")
            trainable_parameters = [p for p in primary_model.parameters() if p.requires_grad]
            ema_model.store(trainable_parameters)
            ema_model.copy_to(trainable_parameters)
            # Save EMA LoRA weights
            transformer_lora_layers = get_peft_model_state_dict(primary_model)
            WanPipelineFixedRef.save_lora_weights(
                os.path.join(output_dir, "ema"),
                transformer_lora_layers=transformer_lora_layers,
                weight_name=f"{cfg.experiment.run_id}.safetensors",
            )
            ema_model.restore(trainable_parameters)

        transformer_lora_layers_to_save = None
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
                if cfg.experiment.use_lora:
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    
                    # Include norm layers if trained
                    if hasattr(cfg.network, 'train_norm_layers') and cfg.network.train_norm_layers:
                        transformer_norm_layers_to_save = {
                            f"transformer.{name}": param
                            for name, param in model.named_parameters()
                            if any(k in name for k in NORM_LAYER_PREFIXES)
                        }
                        transformer_lora_layers_to_save = {
                            **transformer_lora_layers_to_save,
                            **transformer_norm_layers_to_save,
                        }
                    
                    WanPipelineFixedRef.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=transformer_lora_layers_to_save,
                        weight_name=f"{cfg.experiment.run_id}.safetensors",
                    )
                    logger.info(f"Saved LoRA weights to {output_dir}")
                else:
                    model.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

            if weights:
                weights.pop()

    def load_model_hook(models, input_dir):
        if cfg.hparams.ema.use_ema and ema_model is not None:
            logger.info(f"Loading EMA model from Path: {input_dir!r}")
            try:
                ema_model.load_state_dict(os.path.join(input_dir, "ema_model.pt"))
            except Exception as e:
                logger.error(f"Could not load EMA model: {e!r}")

        if cfg.experiment.use_lora:
            # Load from single safetensors file saved by save_model_hook
            lora_weights_path = os.path.join(input_dir, f"{cfg.experiment.run_id}.safetensors")
            
            if os.path.exists(lora_weights_path):
                logger.info(f"Loading LoRA weights from {lora_weights_path}")
                from safetensors.torch import load_file
                lora_state_dict = load_file(lora_weights_path)
                while len(models) > 0:
                    model = models.pop()
                    set_peft_model_state_dict(model, lora_state_dict)
            else:
                logger.warning(f"LoRA weights not found at {lora_weights_path}, skipping load")
                while len(models) > 0:
                    models.pop()
        else:
            # Full model loading
            transformer_ = None
            init_under_meta = False
            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(model, type(unwrap_model(transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                with init_empty_weights():
                    transformer_ = WanTransformer3DModelFixedRef.from_pretrained(
                        cfg.model.pretrained_model_name_or_path,
                        subfolder="transformer"
                    )
                    transformer_.to(accelerator.device, weight_dtype)
                    init_under_meta = True

            load_transformer_model = WanTransformer3DModelFixedRef.from_pretrained(
                input_dir, subfolder="transformer"
            )
            transformer_.register_to_config(**load_transformer_model.config)
            transformer_.load_state_dict(load_transformer_model.state_dict(), assign=init_under_meta)
            del load_transformer_model

        logger.info(f"Completed loading checkpoint from Path: {input_dir!r}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.hparams.max_train_steps is None:
        len_train_dataloader_after_sharding = len(train_dataloader)
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / cfg.hparams.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            cfg.hparams.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = cfg.hparams.max_train_steps * accelerator.num_processes

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler
        lr_scheduler = DummyScheduler(
            name=cfg.hparams.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=num_training_steps_for_scheduler,
            num_warmup_steps=cfg.hparams.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            name=cfg.hparams.lr_scheduler,
            optimizer=optimizer,
            num_training_steps=num_training_steps_for_scheduler,
            num_warmup_steps=cfg.hparams.lr_warmup_steps * accelerator.num_processes,
            num_cycles=cfg.hparams.lr_scheduler_num_cycles,
            power=cfg.hparams.lr_scheduler_power,
        )

    if accelerator.state.deepspeed_plugin is not None:
        d = transformer.config.num_attention_heads * transformer.config.attention_head_dim
        accelerator.state.deepspeed_plugin.deepspeed_config["zero_optimization"]["reduce_bucket_size"] = d
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = cfg.data.batch_size
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ] = cfg.hparams.gradient_accumulation_steps

    transformer, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        transformer, train_dataloader, optimizer, lr_scheduler
    )

    if cfg.hparams.ema.use_ema and ema_model is not None:
        if cfg.hparams.ema.ema_device == "accelerator":
            logger.info("Moving EMA model weights to accelerator...")
        ema_model.to(
            (accelerator.device if cfg.hparams.ema.ema_device == "accelerator" else "cpu"),
            dtype=weight_dtype
        )
        if cfg.hparams.ema.ema_device == "cpu" and not cfg.hparams.ema.ema_cpu_only:
            logger.info("Pinning EMA model weights to CPU...")
            try:
                ema_model.pin_memory()
            except Exception as e:
                logger.error(f"Failed to pin EMA model to CPU: {e}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.hparams.gradient_accumulation_steps)
    if cfg.hparams.max_train_steps is None:
        cfg.hparams.max_train_steps = cfg.hparams.num_train_epochs * num_update_steps_per_epoch

    cfg.hparams.num_train_epochs = math.ceil(cfg.hparams.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = cfg.data.batch_size * accelerator.num_processes * cfg.hparams.gradient_accumulation_steps
    num_trainable_parameters = sum(p.numel() for p in transformer_parameters)

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters ........................................... {num_trainable_parameters}")
    logger.info(f"  Num examples ....................................................... {len(train_dataset)}")
    logger.info(f"  Num batches each epoch ............................................. {len(train_dataloader)}")
    logger.info(f"  Num epochs ......................................................... {cfg.hparams.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device ................................ {cfg.data.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) ... {total_batch_size}")
    logger.info(f"  Gradient accumulation steps ........................................ {cfg.hparams.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps ........................................... {cfg.hparams.max_train_steps}")

    global_step, first_epoch = 0, 0

    if not cfg.checkpointing.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if cfg.checkpointing.resume_from_checkpoint != "latest":
            path = cfg.checkpointing.resume_from_checkpoint
        else:
            dirs = os.listdir(cfg.experiment.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            path = os.path.join(cfg.experiment.output_dir, path)

        if path is None:
            accelerator.print(
                f"Checkpoint {cfg.checkpointing.resume_from_checkpoint!r} does not exist. Starting a new training run."
            )
            cfg.checkpointing.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path!r}")
            accelerator.load_state(path)
            global_step = int(path.split("checkpoint-step")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Override: global_step={initial_global_step} | first_epoch={first_epoch}")

    memory_statistics = get_memory_statistics()
    logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

    # =======================================================
    # 5. training loop
    # =======================================================
    accelerator.wait_for_everyone()
    progress_bar = tqdm(
        range(0, cfg.hparams.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    generator = torch.Generator(device=accelerator.device)
    if cfg.experiment.random_seed is not None:
        # Use rank-offset seed for generator as well
        generator = generator.manual_seed(cfg.experiment.random_seed + accelerator.process_index)
    
    # Track visualization steps to avoid re-saving on resume
    visualization_steps_saved = set()
    if initial_global_step > 0:
        # Mark steps before resume point as already saved
        visualization_steps_saved = set(range(min(5, initial_global_step)))

    # ============================================
    # Visualization helper function
    # ============================================
    def save_training_visualization(batch, global_step, save_dir):
        """Save batch data for visualization to verify data pipeline."""
        import cv2
        from pathlib import Path
        from moviepy import ImageSequenceClip
        
        vis_dir = Path(save_dir) / f"vis_step_{global_step:08d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get first sample from batch
        target = batch["target_images"][0].cpu().numpy()  # [F, C, H, W]
        masked = batch["masked_video"][0].cpu().numpy()   # [F, C, H, W]
        mask_v = batch["mask_video"][0].cpu().numpy()     # [F, 1, H, W]
        ref = batch["ref_image"][0].cpu().numpy()         # [C, H, W]
        mask_i = batch["mask_image"][0].cpu().numpy()     # [1, H, W]
        ref_masked_i = batch["ref_masked_image"][0].cpu().numpy()  # [1, H, W]
        caption = batch["caption"][0] if isinstance(batch["caption"], list) else batch["caption"]
        
        # Denormalize images: [-1,1] -> [0,255]
        def denorm(x):
            return ((x + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # ==================== Save Videos ====================
        # Save target video
        target_frames = [denorm(target[i].transpose(1, 2, 0)) for i in range(target.shape[0])]
        clip = ImageSequenceClip(target_frames, fps=16)
        clip.write_videofile(str(vis_dir / "target_video.mp4"), codec="libx264", logger=None)
        
        # Save masked video
        masked_frames = [denorm(masked[i].transpose(1, 2, 0)) for i in range(masked.shape[0])]
        clip = ImageSequenceClip(masked_frames, fps=16)
        clip.write_videofile(str(vis_dir / "masked_video.mp4"), codec="libx264", logger=None)
        
        # Save mask video (grayscale to RGB for video)
        mask_frames = [(mask_v[i, 0] * 255).clip(0, 255).astype(np.uint8) for i in range(mask_v.shape[0])]
        mask_frames_rgb = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in mask_frames]
        clip = ImageSequenceClip(mask_frames_rgb, fps=16)
        clip.write_videofile(str(vis_dir / "mask_video.mp4"), codec="libx264", logger=None)
        
        # ==================== Save First Frames ====================
        cv2.imwrite(str(vis_dir / "target_frame0.png"), cv2.cvtColor(target_frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(vis_dir / "masked_frame0.png"), cv2.cvtColor(masked_frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(vis_dir / "mask_frame0.png"), mask_frames[0])
        
        # ==================== Save Images ====================
        # Save ref image
        ref_img = denorm(ref.transpose(1, 2, 0))
        cv2.imwrite(str(vis_dir / "ref_image.png"), cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
        
        # Save mask image (first frame mask)
        mask_img_vis = (mask_i[0] * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / "mask_image.png"), mask_img_vis)
        
        # Save ref_masked_image (reference mask)
        ref_masked_img_vis = (ref_masked_i[0] * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / "ref_masked_image.png"), ref_masked_img_vis)
        
        # ==================== Save Caption ====================
        with open(vis_dir / "caption.txt", "w") as f:
            f.write(str(caption))
        
        # ==================== Save Shapes Info ====================
        with open(vis_dir / "shapes.txt", "w") as f:
            f.write(f"target_images: {batch['target_images'].shape}\n")
            f.write(f"masked_video: {batch['masked_video'].shape}\n")
            f.write(f"mask_video: {batch['mask_video'].shape}\n")
            f.write(f"ref_image: {batch['ref_image'].shape}\n")
            f.write(f"mask_image: {batch['mask_image'].shape}\n")
            f.write(f"ref_masked_image: {batch['ref_masked_image'].shape}\n")
            f.write(f"caption: {caption}\n")
        
        logger.info(f"Saved visualization (videos + images) to {vis_dir}")

    for epoch in range(first_epoch, cfg.hparams.num_train_epochs):
        logger.info(f"epoch {epoch+1}/{cfg.hparams.num_train_epochs}")
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            # ============================================
            # Save visualization for first few steps (skip if already saved on resume)
            # ============================================
            if accelerator.is_main_process and global_step < 5 and global_step not in visualization_steps_saved:
                save_training_visualization(batch, global_step, output_dirpath / "visualizations")
                visualization_steps_saved.add(global_step)
            
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                # ============================================
                # Prepare inputs
                # ============================================
                # Ground truth video: [B, F, C, H, W] -> [B, C, F, H, W]
                video = rearrange(batch["target_images"], "b f c h w -> b c f h w").to(dtype=weight_dtype)
                
                # Masked video (mask==1 region blacked out): [B, F, C, H, W] -> [B, C, F, H, W]
                masked_video = rearrange(batch["masked_video"], "b f c h w -> b c f h w").to(dtype=weight_dtype)
                
                # Mask video: [B, F, 1, H, W] -> [B*F, 1, H, W]
                mask_video = rearrange(batch["mask_video"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                
                # Reference image (first frame foreground): [B, C, H, W]
                ref_img = batch["ref_image"].to(dtype=weight_dtype)
                
                # Mask image (first frame mask): [B, 1, H, W]
                mask_img = batch["ref_masked_image"].to(dtype=weight_dtype)
                
                # Caption
                prompt = batch["caption"]

                # ============================================
                # Encode to latent space
                # ============================================
                with torch.no_grad():
                    latents = prepare_latents(vae, video, device=accelerator.device, dtype=weight_dtype)
                    masked_video_latents = prepare_latents(vae, masked_video, device=accelerator.device, dtype=weight_dtype)
                    
                    # Encode reference image (expand to video format first)
                    # ref_img: [B, C, H, W] -> [B, C, 1, H, W]
                    ref_img_video = ref_img.unsqueeze(2)
                    ref_latents = prepare_latents(vae, ref_img_video, device=accelerator.device, dtype=weight_dtype)

                    # Prepare mask video at latent size
                    mask_lat_size = prepare_mask_latent_size(mask_video, cfg.data.nframes) # 4通道 -> [B 4 f h w]
                    num_latent_frames = mask_lat_size.shape[2]
                    
                    # ref_mask 尺度 -> [B, 4, 1, h, w], mask_video -> [B, 4, F, h, w]
                    # ref_latents 尺度 -> [B, 16, 1, h, w]
                    mask_img_lat_size = F.interpolate(mask_img, scale_factor=1/16, mode="nearest-exact")  # [B, 1, h, w]
                    mask_img_lat_size = mask_img_lat_size.unsqueeze(2).repeat(1, 4, 1, 1, 1)  # [B, 4, 1, h, w]
                    
                    prompt_embeds = get_t5_prompt_embeds(
                        text_encoder, tokenizer, prompt,
                        device=latents.device, dtype=weight_dtype
                    )

                batch_size = latents.size(0)

                # ============================================
                # Add noise to ground truth
                # ============================================
                noise = torch.randn(latents.shape, device=accelerator.device, dtype=weight_dtype, generator=generator)
                timestep_id = torch.randint(0, noise_scheduler.num_train_timesteps, (1,))
                timestep = noise_scheduler.timesteps[timestep_id].to(dtype=weight_dtype, device=accelerator.device)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timestep)
                training_target = noise_scheduler.training_target(latents, noise, timestep)
                
                zeros_latents = torch.zeros_like(latents)  # [B, 48, F, h, w]
                zeros_latents_4ch = torch.zeros_like(mask_lat_size)  # [B, 4, F, h, w]

                # ============================================
                # Three-branch temporal fusion + channel fusion
                # 100-channel structure (48 + 4 + 48 = 100)
                # ============================================
                
                # Runtime shape validation before concatenation
                expected_temporal_frames = num_latent_frames * 2 + ref_latents.shape[2]  # 2F + 1
                assert noisy_model_input.shape[2] == num_latent_frames, \
                    f"noisy_model_input temporal dim mismatch: {noisy_model_input.shape[2]} vs {num_latent_frames}"
                assert masked_video_latents.shape[2] == num_latent_frames, \
                    f"masked_video_latents temporal dim mismatch: {masked_video_latents.shape[2]} vs {num_latent_frames}"
                assert mask_lat_size.shape[2] == num_latent_frames, \
                    f"mask_lat_size temporal dim mismatch: {mask_lat_size.shape[2]} vs {num_latent_frames}"
                assert ref_latents.shape[2] == 1, \
                    f"ref_latents should have 1 frame, got {ref_latents.shape[2]}"
                assert mask_img_lat_size.shape[2] == 1, \
                    f"mask_img_lat_size should have 1 frame, got {mask_img_lat_size.shape[2]}"
                
                # Verify spatial dimensions match
                H_lat, W_lat = latents.shape[3], latents.shape[4]
                assert ref_latents.shape[3:] == (H_lat, W_lat), \
                    f"ref_latents spatial dim mismatch: {ref_latents.shape[3:]} vs ({H_lat}, {W_lat})"
                
                branch1 = torch.cat([noisy_model_input, masked_video_latents, ref_latents], dim=2)  # [B, 16, 2F+1, h, w]
                branch2 = torch.cat([zeros_latents_4ch, mask_lat_size, mask_img_lat_size], dim=2)   # [B, 4, 2F+1, h, w]
                branch3 = torch.cat([zeros_latents, masked_video_latents, ref_latents], dim=2)      # [B, 16, 2F+1, h, w]
                model_input = torch.cat([branch1, branch2, branch3], dim=1)
                
                # Release intermediate tensors to free memory
                del branch1, branch2, branch3, zeros_latents, zeros_latents_4ch

                # Debug shape verification (first step only)
                if step == 0 and accelerator.is_main_process:
                    logger.info(f"Model input shape: {model_input.shape}")
                    logger.info(f"Expected: [B, 36, {expected_temporal_frames}, H, W]")

                num_latent_frames = latents.shape[2]
                ref_num_frames = ref_latents.shape[2]
                frame_segments = [
                    (num_latent_frames, False),   # noisy_gt: sequential indices 0 to F-1
                    (num_latent_frames, False),   # masked_video: sequential indices F to 2F-1
                    (ref_num_frames, True),       # ref_img: FIXED position 60
                ]

                denoised_latents_full = transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=None,
                    frame_segments=frame_segments,
                    return_dict=False,
                )[0]
                
                # Extract only the first F frames (corresponding to noisy_gt prediction)
                # denoised_latents_full: [B, 16, 3F, H, W] -> denoised_latents: [B, 16, F, H, W]
                num_latent_frames = latents.shape[2]  # F
                denoised_latents = denoised_latents_full[:, :, :num_latent_frames, :, :]
                
                if step == 0 and accelerator.is_main_process:
                    logger.info(f"Transformer output shape: {denoised_latents_full.shape}")
                    logger.info(f"Extracted denoised_latents shape: {denoised_latents.shape}")
                    logger.info(f"Training target shape: {training_target.shape}")

                # ============================================
                # Compute loss
                # ============================================
                weighting = noise_scheduler.training_weight(timestep)
                weight_mask = mask_lat_size[:, 0:1]  # [B, 1, F, H, W]
                
                # Release large intermediate tensors before loss computation
                del denoised_latents_full, model_input
                
                # Weighted MSE loss with mask emphasis
                loss = torch.mean(
                    (weighting * (denoised_latents.float() - training_target.float()) ** 2 * (1 + weight_mask.float())).reshape(training_target.shape[0], -1),
                    1,
                )
                assert torch.isnan(loss).sum() == 0, "NaN loss detected"
                
                # Release more intermediate tensors
                del denoised_latents, training_target, noisy_model_input

                accelerator.backward(loss)
                
                if cfg.hparams.gradient_precision == "fp32":
                    for param in transformer_parameters:
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(torch.float32)

                grad_norm = max_gradient(transformer_parameters)

                if accelerator.sync_gradients:
                    if accelerator.distributed_type == DistributedType.DEEPSPEED:
                        grad_norm = transformer.get_global_grad_norm()
                    elif cfg.hparams.max_grad_norm > 0:
                        if cfg.hparams.grad_clip_method == "norm":
                            grad_norm = accelerator.clip_grad_norm_(
                                transformer_parameters, cfg.hparams.max_grad_norm
                            )
                        elif cfg.hparams.grad_clip_method == "value":
                            grad_norm = accelerator.clip_grad_value_(
                                transformer_parameters, cfg.hparams.max_grad_norm
                            )
                
                if torch.is_tensor(grad_norm):
                    grad_norm = grad_norm.item()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if cfg.hparams.ema.use_ema and ema_model is not None:
                    ema_model.step(parameters=transformer_parameters, global_step=global_step)
                
                if accelerator.is_main_process:
                    if global_step % cfg.checkpointing.save_every_n_steps == 0:
                        save_path = os.path.join(output_dirpath, f"checkpoint-step{global_step:08d}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path!r}")

                        if cfg.checkpointing.save_last_n_steps is not None:
                            remove_step_no = global_step - cfg.checkpointing.save_last_n_steps - 1
                            remove_step_no = remove_step_no - (remove_step_no % cfg.checkpointing.save_every_n_steps)
                            if remove_step_no < 0:
                                remove_step_no = None
                            if remove_step_no is not None:
                                remove_ckpt_name = os.path.join(output_dirpath, f"checkpoint-step{remove_step_no:08d}")
                                if os.path.exists(remove_ckpt_name):
                                    logger.info(f"removing old checkpoint: {remove_ckpt_name!r}")
                                    shutil.rmtree(remove_ckpt_name)

            logs = {}
            logs["loss"] = accelerator.reduce(loss.detach().clone(), reduction="mean").item()
            logs["grad_norm"] = grad_norm
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            if ema_model is not None:
                logs["ema_decay"] = ema_model.get_decay()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            # Memory cleanup - ALL processes clear cache for distributed training
            if global_step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()  # Also trigger Python garbage collection

            if global_step >= cfg.hparams.max_train_steps:
                logger.info(f"max training steps={cfg.hparams.max_train_steps!r} reached.")
                break

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.hparams.ema.use_ema and ema_model is not None:
            ema_model.copy_to(transformer_parameters)

        transformer = unwrap_model(transformer)
        
        if cfg.experiment.use_lora:
            # Save final LoRA weights using get_peft_model_state_dict
            transformer_lora_layers = get_peft_model_state_dict(transformer)
            
            # Include norm layers if trained
            if hasattr(cfg.network, 'train_norm_layers') and cfg.network.train_norm_layers:
                transformer_norm_layers = {
                    f"transformer.{name}": param
                    for name, param in transformer.named_parameters()
                    if any(k in name for k in NORM_LAYER_PREFIXES)
                }
                transformer_lora_layers = {
                    **transformer_lora_layers,
                    **transformer_norm_layers,
                }
            
            WanPipelineFixedRef.save_lora_weights(
                output_dirpath,
                transformer_lora_layers=transformer_lora_layers,
                safe_serialization=True,
                weight_name=f"{cfg.experiment.run_id}.safetensors",
            )
            logger.info(f"Saved final LoRA weights to {output_dirpath}")
        else:
            transformer.save_pretrained(output_dirpath)
    
    accelerator.wait_for_everyone()

    memory_statistics = get_memory_statistics()
    logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
