"""
S2V-Style Video Inpainting Training Script

Key features:
- Model input on channel: noisy_latent(16) + masked_video(16) + mask(16) = 48 channels
- Reference concatenated on temporal (d) dimension like S2V
- Reference RoPE position fixed at 30
- patch_embedding: 48 channels

Architecture:
    model_input = concat([noisy_latent, masked_video_latent, mask_latent], dim=1)  # 48ch on channel
    Transformer receives:
        - hidden_states: [B, 48, F, H, W]
        - ref_latents: [B, 16, 1, H, W] -> concatenated on sequence dimension inside transformer
    Loss computed on denoised output (only video frames, not ref)
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
from models.transformer_wan_s2v_style import WanTransformer3DModelS2VStyle
from models.autoencoder_kl_wan import AutoencoderKLWan
from models.flow_match import FlowMatchScheduler
from pipelines.pipeline_wan_inpainting_s2v_style import WanPipelineS2VStyle, retrieve_latents, prompt_clean
from ema import EMAModel


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


def prepare_mask_latent_size(mask_values: torch.Tensor, nframes: int, num_channels: int = 4) -> torch.Tensor:
    """
    Prepare mask for latent space size.
    
    Args:
        mask_values: [B*F, 1, H, W] mask values
        nframes: number of frames
        num_channels: number of output channels (default 4 for 100ch input: 48+48+4)
        
    Returns:
        mask_latents: [B, num_channels, F_lat, H_lat, W_lat] mask expanded to num_channels
    """
    latent_masks = rearrange(
        F.interpolate(mask_values, scale_factor=1/16, mode="nearest-exact"),
        "(b f) c h w -> b c f h w", f=nframes
    )
    
    # Handle temporal compression (4x)
    # First frame is repeated 4 times, then rest of frames
    first_frame_mask = latent_masks[:, :, 0:1]
    first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=4)
    mask_latents = torch.concat([first_frame_mask, latent_masks[:, :, 1:, :]], dim=2)
    
    batch_size, _, _, latent_height, latent_width = mask_latents.shape
    num_latent_frames = (nframes - 1) // 4 + 1
    
    # Reshape to get correct latent temporal dimension
    mask_latents = mask_latents[:, :, :num_latent_frames*4, :, :]
    mask_latents = mask_latents.view(batch_size, -1, num_latent_frames, 4, latent_height, latent_width)
    mask_latents = mask_latents[:, :, :, 0, :, :]  # Take first of every 4
    
    # Expand to num_channels (4 for Wan2.2: 48+48+4=100)
    mask_latents = mask_latents.repeat(1, num_channels, 1, 1, 1)
    
    return mask_latents


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

    print(accelerator.state)
    accelerator.print("\nENVIRONMENT\n")
    accelerator.print(f"  Python .......................... {sys.version}")
    accelerator.print(f"  torch.__version__ ............... {torch.__version__}")
    accelerator.print(f"  torch.version.cuda .............. {torch.version.cuda}")
    accelerator.print(f"  torch.backends.cudnn.version() .. {torch.backends.cudnn.version()}\n")
    accelerator.print(f">> Run ID : {cfg.experiment.run_id!r}")
    accelerator.print(">> S2V-Style Training: 48ch input, ref on temporal dim, RoPE@30")

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

    if accelerator.is_main_process:
        output_dirpath.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        logger.info(f"Saving config to {output_dirpath / 'config.yaml'}")
        yaml_cfg = pyrallis.dump(cfg)
        with open(output_dirpath / "config.yaml", "w") as f:
            f.write(yaml_cfg)

    logger.info(f"config = \n{pyrallis.dump(cfg)}")

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
        low_cpu_mem_usage=False,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=load_dtype,
        low_cpu_mem_usage=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    
    logger.info(f"Load base transformer from {cfg.model.pretrained_model_transformer_name_or_path!r}")

    # Load base transformer and create S2V-style model
    # patch_embedding: 100ch, ref_patch_embedding: 48ch
    transformer = WanTransformer3DModelS2VStyle.from_pretrained(
        cfg.model.pretrained_model_transformer_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        fixed_ref_position=30,
        low_cpu_mem_usage=False, 
        device_map=None,
    )
    
    logger.info(f"Transformer config: in_channels={transformer.config.in_channels}, out_channels={transformer.config.out_channels}")
    logger.info(f"Fixed ref RoPE position: {transformer.config.fixed_ref_position}")
    logger.info("Initializing new layers: ref_patch_embedding, trainable_cond_mask...")
    
    with torch.no_grad():
        if transformer.patch_embedding.weight.shape == transformer.ref_patch_embedding.weight.shape:
            transformer.ref_patch_embedding.weight.copy_(transformer.patch_embedding.weight)
            if transformer.patch_embedding.bias is not None and transformer.ref_patch_embedding.bias is not None:
                transformer.ref_patch_embedding.bias.copy_(transformer.patch_embedding.bias)
            logger.info("Copied patch_embedding weights to ref_patch_embedding")
        else:
            # Use kaiming initialization if shapes don't match
            nn.init.kaiming_normal_(transformer.ref_patch_embedding.weight, mode='fan_out', nonlinearity='linear')
            if transformer.ref_patch_embedding.bias is not None:
                nn.init.zeros_(transformer.ref_patch_embedding.bias)
            logger.info(f"Kaiming initialized ref_patch_embedding (shape mismatch: {transformer.patch_embedding.weight.shape} vs {transformer.ref_patch_embedding.weight.shape})")
        
        # Initialize trainable_cond_mask embedding with small values
        nn.init.normal_(transformer.trainable_cond_mask.weight, mean=0.0, std=0.02)
        logger.info("Initialized trainable_cond_mask with normal distribution (std=0.02)")

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
        transformer.requires_grad_(False)
        
        lora_config = LoraConfig(
            r=cfg.network.lora_rank,
            lora_alpha=cfg.network.lora_alpha,
            target_modules=cfg.network.target_modules or ["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=cfg.network.lora_dropout,
            init_lora_weights=cfg.network.init_lora_weights,
        )
        
        transformer = get_peft_model(transformer, lora_config)
        logger.info(f"LoRA config: rank={cfg.network.lora_rank}, alpha={cfg.network.lora_alpha}")
        
        # Enable gradients for new layers (ref_patch_embedding, trainable_cond_mask)
        # These are not LoRA layers, but need to be trained from scratch
        for name, param in transformer.named_parameters():
            if "ref_patch_embedding" in name or "trainable_cond_mask" in name:
                param.requires_grad = True
                logger.info(f"Enabled training for new layer: {name}")
        
        transformer.print_trainable_parameters()
        
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
    from dataloading.dataset_inpaint_s2v_style import InpaintingDatasetS2VStyle
    
    debug_save_dir = output_dirpath / "debug_data"

    logger.info("Using S2V-style InpaintingDataset (no ref augmentation, maximize mask region)")
    train_dataset = InpaintingDatasetS2VStyle(
        cfg.data,
        save_debug=True,
        debug_output_dir=str(debug_save_dir),
        debug_save_prob=0.001,
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

        transformer_lora_layers_to_save = None
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
                if cfg.experiment.use_lora:
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    WanPipelineS2VStyle.save_lora_weights(
                        output_dir,
                        transformer_lora_layers=transformer_lora_layers_to_save,
                        weight_name=f"{cfg.experiment.run_id}.safetensors",
                    )
                    logger.info(f"Saved LoRA weights to {output_dir}")
                    
                    # Save new layers (ref_patch_embedding, trainable_cond_mask) separately
                    # These are full layers, not LoRA, so save them as a separate file
                    from safetensors.torch import save_file
                    new_layers_state_dict = {
                        "ref_patch_embedding.weight": model.ref_patch_embedding.weight.data.clone(),
                        "trainable_cond_mask.weight": model.trainable_cond_mask.weight.data.clone(),
                    }
                    if model.ref_patch_embedding.bias is not None:
                        new_layers_state_dict["ref_patch_embedding.bias"] = model.ref_patch_embedding.bias.data.clone()
                    
                    new_layers_path = os.path.join(output_dir, "s2v_new_layers.safetensors")
                    save_file(new_layers_state_dict, new_layers_path)
                    logger.info(f"Saved new layers (ref_patch_embedding, trainable_cond_mask) to {new_layers_path}")
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
            lora_weights_path = os.path.join(input_dir, f"{cfg.experiment.run_id}.safetensors")
            new_layers_path = os.path.join(input_dir, "s2v_new_layers.safetensors")
            
            if os.path.exists(lora_weights_path):
                logger.info(f"Loading LoRA weights from {lora_weights_path}")
                from safetensors.torch import load_file
                lora_state_dict = load_file(lora_weights_path)
                while len(models) > 0:
                    model = models.pop()
                    set_peft_model_state_dict(model, lora_state_dict)
                    
                    # Load new layers (ref_patch_embedding, trainable_cond_mask)
                    if os.path.exists(new_layers_path):
                        logger.info(f"Loading new layers from {new_layers_path}")
                        new_layers_dict = load_file(new_layers_path)
                        unwrapped = unwrap_model(model)
                        if "ref_patch_embedding.weight" in new_layers_dict:
                            unwrapped.ref_patch_embedding.weight.data.copy_(new_layers_dict["ref_patch_embedding.weight"])
                        if "ref_patch_embedding.bias" in new_layers_dict:
                            unwrapped.ref_patch_embedding.bias.data.copy_(new_layers_dict["ref_patch_embedding.bias"])
                        if "trainable_cond_mask.weight" in new_layers_dict:
                            unwrapped.trainable_cond_mask.weight.data.copy_(new_layers_dict["trainable_cond_mask.weight"])
                        logger.info("Loaded new layers (ref_patch_embedding, trainable_cond_mask)")
                    else:
                        logger.warning(f"New layers file not found at {new_layers_path}")
            else:
                logger.warning(f"LoRA weights not found at {lora_weights_path}, skipping load")
                while len(models) > 0:
                    models.pop()
        else:
            transformer_ = None
            while len(models) > 0:
                model = models.pop()
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_ = model
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

            load_transformer_model = WanTransformer3DModelS2VStyle.from_pretrained(
                input_dir, subfolder="transformer"
            )
            transformer_.register_to_config(**load_transformer_model.config)
            transformer_.load_state_dict(load_transformer_model.state_dict())
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

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.hparams.gradient_accumulation_steps)
    if cfg.hparams.max_train_steps is None:
        cfg.hparams.max_train_steps = cfg.hparams.num_train_epochs * num_update_steps_per_epoch

    cfg.hparams.num_train_epochs = math.ceil(cfg.hparams.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = cfg.data.batch_size * accelerator.num_processes * cfg.hparams.gradient_accumulation_steps
    num_trainable_parameters = sum(p.numel() for p in transformer_parameters)

    logger.info("***** Running S2V-style training *****")
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
        generator = generator.manual_seed(cfg.experiment.random_seed + accelerator.process_index)

    for epoch in range(first_epoch, cfg.hparams.num_train_epochs):
        logger.info(f"epoch {epoch+1}/{cfg.hparams.num_train_epochs}")
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                # ============================================
                # Prepare inputs (S2V-style)
                # ============================================
                # Ground truth video: [B, F, C, H, W] -> [B, C, F, H, W]
                video = rearrange(batch["target_images"], "b f c h w -> b c f h w").to(dtype=weight_dtype)
                
                # Masked video (mask==1 region blacked out): [B, F, C, H, W] -> [B, C, F, H, W]
                masked_video = rearrange(batch["masked_video"], "b f c h w -> b c f h w").to(dtype=weight_dtype)
                
                # Mask video: [B, F, 1, H, W] -> [B*F, 1, H, W]
                mask_video = rearrange(batch["mask_video"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                
                # Reference image (NO augmentation): [B, C, H, W]
                ref_img = batch["ref_image"].to(dtype=weight_dtype)
                
                # Caption
                prompt = batch["caption"]

                # ============================================
                # Encode to latent space
                # ============================================
                with torch.no_grad():
                    # Encode GT video
                    latents = prepare_latents(vae, video, device=accelerator.device, dtype=weight_dtype)
                    
                    # Encode masked video
                    masked_video_latents = prepare_latents(vae, masked_video, device=accelerator.device, dtype=weight_dtype)
                    
                    # Encode reference image: [B, C, H, W] -> [B, C, 1, H, W] -> encode -> [B, 16, 1, H_lat, W_lat]
                    ref_img_video = ref_img.unsqueeze(2)
                    ref_latents = prepare_latents(vae, ref_img_video, device=accelerator.device, dtype=weight_dtype)

                    # Prepare mask at latent size: [B, 16, F_lat, H_lat, W_lat]
                    mask_latents = prepare_mask_latent_size(mask_video, cfg.data.nframes)
                    mask_latents = mask_latents.to(dtype=weight_dtype, device=accelerator.device)
                    
                    # Get text embeddings
                    prompt_embeds = get_t5_prompt_embeds(
                        text_encoder, tokenizer, prompt,
                        device=latents.device, dtype=weight_dtype
                    )

                batch_size = latents.size(0)
                num_latent_frames = latents.shape[2]

                # ============================================
                # Add noise to ground truth
                # ============================================
                noise = torch.randn(latents.shape, device=accelerator.device, dtype=weight_dtype, generator=generator)
                timestep_id = torch.randint(0, noise_scheduler.num_train_timesteps, (1,))
                timestep = noise_scheduler.timesteps[timestep_id].to(dtype=weight_dtype, device=accelerator.device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)
                training_target = noise_scheduler.training_target(latents, noise, timestep)

                # ============================================
                # Build 100-channel model input (Wan2.2 S2V style)
                # model_input = concat([noisy_latent(48), masked_video(48), mask(4)], dim=1)
                # ============================================
                # Concat on channel dim: noisy(48) + masked_video(48) + mask(4) = 100 channels
                model_input = torch.cat([noisy_latents, masked_video_latents, mask_latents], dim=1)
                
                if step == 0 and accelerator.is_main_process:
                    logger.info(f"S2V-style model input shape: {model_input.shape} (expected [B, 100, F, H, W])")
                    logger.info(f"Reference latents shape: {ref_latents.shape} (expected [B, 48, 1, H, W])")

                # ============================================
                # Forward pass (ref concatenated on d inside transformer)
                # ============================================
                denoised_latents = transformer(
                    hidden_states=model_input,
                    ref_latents=ref_latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
                
                if step == 0 and accelerator.is_main_process:
                    logger.info(f"Transformer output shape: {denoised_latents.shape}")
                    logger.info(f"Training target shape: {training_target.shape}")

                # ============================================
                # Compute loss
                # ============================================
                weighting = noise_scheduler.training_weight(timestep)
                weight_mask = mask_latents[:, 0:1]  # [B, 1, F, H, W]
                
                # Weighted MSE loss with mask emphasis
                loss = torch.mean(
                    (weighting * (denoised_latents.float() - training_target.float()) ** 2 * (1 + weight_mask.float())).reshape(training_target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                assert torch.isnan(loss).sum() == 0, "NaN loss detected"

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
            
            if global_step % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

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
            transformer_lora_layers = get_peft_model_state_dict(transformer)
            WanPipelineS2VStyle.save_lora_weights(
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
