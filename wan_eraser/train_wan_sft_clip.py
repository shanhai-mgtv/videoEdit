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
import torchvision.transforms.v2 as transforms
from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from einops import rearrange, repeat
from PIL import Image
from decord import VideoReader
from tqdm.auto import tqdm
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from moviepy import ImageSequenceClip

from peft import LoraConfig, get_peft_model

from config import Config  # isort:skip
from optim import get_optimizer, max_gradient  # isort: skip
from dataloading.dataset_clip import CLIPDataset
from dataloading import utils
from models.transformer_wan import WanTransformer3DModel
from models.autoencoder_kl_wan import AutoencoderKLWan
from models.flow_match import FlowMatchScheduler
from pipelines.pipeline_wan_inpainting import WanPipeline, retrieve_latents, prompt_clean
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
        return np.array([])

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if mask:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = (frame_gray > 5).astype(np.uint8)*255
            frame_gray = frame_gray[None, :, :]
            frames.append(frame_gray)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()

    return np.array(frames), fps


def get_memory_statistics():
    return {
        f"cuda:{i}": {
            "allocated": f"{torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB",
            "reserved": f"{torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB",
        }
        for i in range(torch.cuda.device_count())
    }


def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            if param.requires_grad:
                param.data = param.to(dtype)


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


class CLIPToTextProjection(nn.Module):
    """
    Project CLIP image embeddings to text embedding dimension for cross attention.
    CLIP ViT-Base-Patch32 output: 768 -> UMT5 text encoder output: 4096
    """
    def __init__(self, clip_dim: int = 768, text_dim: int = 4096):
        super().__init__()
        self.proj = nn.Linear(clip_dim, text_dim)
        self.norm = nn.LayerNorm(text_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


def get_clip_image_embeds(
    image_encoder: CLIPVisionModel,
    image_processor: CLIPImageProcessor,
    image: torch.Tensor,
    clip_proj: CLIPToTextProjection,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Encode image using CLIP image encoder to get embeddings for cross attention.
    
    Args:
        image_encoder: CLIPVisionModel
        image_processor: CLIPImageProcessor
        image: [B, C, H, W] tensor in range [-1, 1] or [0, 1]
        clip_proj: CLIPToTextProjection to project CLIP dim to text dim
        device: target device
        dtype: target dtype
        
    Returns:
        image_embeds: [B, seq_len, text_dim] tensor for cross attention
    """
    device = device or image_encoder.device
    dtype = dtype or image_encoder.dtype
    
    batch_size = image.shape[0]
    
    if image.min() < 0:
        image = (image + 1) / 2  # [-1, 1] -> [0, 1]
    image = (image * 255).clamp(0, 255).to(torch.uint8)
    
    pil_images = []
    for i in range(batch_size):
        img_np = image[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
        pil_img = Image.fromarray(img_np)
        pil_images.append(pil_img)
    
    # Process images with CLIPImageProcessor
    inputs = image_processor(images=pil_images, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device=device, dtype=dtype)
    
    # Get image embeddings from CLIP
    with torch.no_grad():
        image_outputs = image_encoder(pixel_values, output_hidden_states=True)
        image_embeds = image_outputs.hidden_states[-2]  # [B, 50, 768] for ViT-Base-Patch32 (224/32=7, 7*7+1=50 tokens)
    
    # Project to text dimension (768 -> 4096)
    image_embeds = clip_proj(image_embeds.to(dtype=dtype))
    
    return image_embeds


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
    accelerator.print("\n")
    accelerator.print(f">> Run ID : {cfg.experiment.run_id!r}")

    if accelerator.is_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.experiment.random_seed is not None:
        set_seed(cfg.experiment.random_seed)
    
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
    
    # Load CLIP image encoder from specified path
    clip_model_path = "/mnt/shanhai-ai/shanhai-workspace/lihaoran/ckps/clip-vit-base-patch32"
    logger.info(f"Loading CLIP image encoder from {clip_model_path}")
    image_encoder = CLIPVisionModel.from_pretrained(
        clip_model_path,
        torch_dtype=load_dtype,
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        clip_model_path,
    )
    logger.info(f"CLIP image encoder loaded successfully")
    
    # Create projection layer to map CLIP dim (768) to text dim (4096)
    # CLIP ViT-Base-Patch32 has hidden_size=768
    clip_proj = CLIPToTextProjection(clip_dim=768, text_dim=4096)
    logger.info(f"Created CLIP to text projection layer (768 -> 4096)")
    
    logger.info(f"Load transformer model from {cfg.model.pretrained_model_name_or_path!r}")

    transformer = WanTransformer3DModel.from_pretrained(
        cfg.model.pretrained_model_transformer_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    logger.info(f"Loaded transformer model from {cfg.model.pretrained_model_name_or_path!r}")

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

    if cfg.hparams.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    clip_proj.requires_grad_(True)

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_proj.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"dit dtype: {next(transformer.parameters()).dtype!r}")

    trainable_params, all_param = get_nb_trainable_parameters(transformer)
    clip_proj_params = sum(p.numel() for p in clip_proj.parameters())
    logger.info(f"clip_proj trainable params: {clip_proj_params:,d}")
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

    if cfg.hparams.mixed_precision == "fp16":
        logger.warning("full fp16 training is unstable, casting params to fp32")
        cast_training_params([transformer])

    if cfg.hparams.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Combine transformer and clip_proj parameters for training
    transformer_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    clip_proj_parameters = list(clip_proj.parameters())
    all_trainable_parameters = transformer_parameters + clip_proj_parameters
    logger.info(f"Total trainable parameters: transformer={len(transformer_parameters)}, clip_proj={len(clip_proj_parameters)}")

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
        all_trainable_parameters,  # Include both transformer and clip_proj parameters
        optimizer_name=cfg.hparams.optimizer_type,
        learning_rate=cfg.hparams.learning_rate,
        optimizer_args_str=cfg.hparams.optimizer_args,
        use_deepspeed=use_deepspeed_optimizer,
    )

    # ======================================================
    # 3. build dataset and dataloaders
    # ======================================================
    train_dataset = CLIPDataset(cfg.data)
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
           
            ema_model.save_pretrained(os.path.join(output_dir, "ema"))
            ema_model.restore(trainable_parameters)
        
        
        for model in models:
            model_unwrapped = unwrap_model(model)
            # Check if it's transformer (handle both PeftModel and base model)
            is_transformer = False
            try:
                if hasattr(model_unwrapped, 'base_model'):  # PeftModel
                    is_transformer = True
                elif isinstance(model_unwrapped, type(unwrap_model(transformer))):
                    is_transformer = True
            except:
                pass
            
            if is_transformer:
                if cfg.experiment.use_lora:
                    # Save LoRA adapter only
                    lora_save_path = os.path.join(output_dir, "lora_adapter")
                    logger.info(f"Saving LoRA adapter to {lora_save_path}")
                    model_unwrapped.save_pretrained(lora_save_path)
                else:
                    # Save full model
                    model_unwrapped.save_pretrained(
                        os.path.join(output_dir, "transformer"), safe_serialization=True, max_shard_size="5GB"
                    )
            elif isinstance(model_unwrapped, CLIPToTextProjection):
                # Save clip_proj state dict
                clip_proj_path = os.path.join(output_dir, "clip_proj.pt")
                torch.save(model_unwrapped.state_dict(), clip_proj_path)
                logger.info(f"Saved clip_proj to {clip_proj_path!r}")
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

        # Handle LoRA loading
        if cfg.experiment.use_lora:
            lora_path = os.path.join(input_dir, "lora_adapter")
            if os.path.exists(lora_path):
                logger.info(f"Loading LoRA adapter from {lora_path}")
                while len(models) > 0:
                    model = models.pop()
                    model_unwrapped = unwrap_model(model)
                    if hasattr(model_unwrapped, 'load_adapter'):
                        model_unwrapped.load_adapter(lora_path, adapter_name="default")
                    elif isinstance(model_unwrapped, CLIPToTextProjection):
                        pass  # Will load clip_proj below
            else:
                logger.warning(f"LoRA adapter not found at {lora_path}, skipping load")
                while len(models) > 0:
                    models.pop()
        else:
            # Full model loading
            transformer_ = None
            clip_proj_ = None
            init_under_meta = False
            
            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    model_unwrapped = unwrap_model(model)
                    if isinstance(model_unwrapped, type(unwrap_model(transformer))):
                        transformer_ = model
                    elif isinstance(model_unwrapped, CLIPToTextProjection):
                        clip_proj_ = model
                    else:
                        raise ValueError(f"unexpected load model: {model.__class__}")
            else:
                with init_empty_weights():
                    transformer_ = WanTransformer3DModel.from_pretrained(
                        cfg.model.pretrained_model_transformer_name_or_path,
                        subfolder="transformer"
                    )
                    transformer_.to(accelerator.device, weight_dtype)
                    init_under_meta = True

            # Load transformer
            if transformer_ is not None:
                load_transformer_model = WanTransformer3DModel.from_pretrained(
                    input_dir, subfolder="transformer"
                )
                unwrap_model(transformer_).register_to_config(**load_transformer_model.config)
                unwrap_model(transformer_).load_state_dict(load_transformer_model.state_dict(), assign=init_under_meta)
                del load_transformer_model

            if cfg.hparams.mixed_precision == "fp16":
                cast_training_params([transformer_])

        # Load clip_proj state dict (for both LoRA and full model)
        clip_proj_path = os.path.join(input_dir, "clip_proj.pt")
        if os.path.exists(clip_proj_path):
            clip_proj.load_state_dict(torch.load(clip_proj_path, map_location=accelerator.device))
            logger.info(f"Loaded clip_proj from {clip_proj_path!r}")
        else:
            logger.warning(f"clip_proj checkpoint not found at {clip_proj_path!r}, using random initialization")

        logger.info(f"Completed loading checkpoint from Path: {input_dir!r}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.hparams.max_train_steps is None:
        len_train_dataloader_after_sharding = len(train_dataloader)
        print("len_train_dataloader_after_sharding: ", len_train_dataloader_after_sharding)
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
    
    # Prepare everything with our `accelerator`.
    
    transformer, clip_proj, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        transformer, clip_proj, train_dataloader, optimizer, lr_scheduler
    )


    if cfg.hparams.ema.use_ema and ema_model is not None:
        if cfg.hparams.ema.ema_device == "accelerator":
            logger.info("Moving EMA model weights to accelerator...")

        ema_model.to((accelerator.device if cfg.hparams.ema.ema_device == "accelerator" else "cpu"), dtype=weight_dtype)

        if cfg.hparams.ema.ema_device == "cpu" and not cfg.hparams.ema.ema_cpu_only:
            logger.info("Pinning EMA model weights to CPU...")
            try:
                ema_model.pin_memory()
            except Exception as e:
                logger.error(f"Failed to pin EMA model to CPU: {e}")


    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.hparams.gradient_accumulation_steps)
    if cfg.hparams.max_train_steps is None:
        cfg.hparams.max_train_steps = cfg.hparams.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != cfg.hparams.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    
    # Afterwards we recalculate our number of training epochs
    cfg.hparams.num_train_epochs = math.ceil(cfg.hparams.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = cfg.data.batch_size * accelerator.num_processes * cfg.hparams.gradient_accumulation_steps
    num_trainable_parameters = sum(p.numel() for p in all_trainable_parameters)

    # fmt: off
    logger.info("***** Running training (CLIP image conditioning) *****")
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
            # Get the most recent checkpoint
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
        generator = generator.manual_seed(cfg.experiment.random_seed)
    
    for epoch in range(first_epoch, cfg.hparams.num_train_epochs):
        logger.info(f"epoch {epoch+1}/{ cfg.hparams.num_train_epochs}")
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                # input video
                video = rearrange(batch["target_images"], "b f c h w -> b c f h w").to(dtype=weight_dtype)  # b c f h w
                
                orig_h, orig_w = video.shape[-2:]
                # input video with mask
                cond_values = rearrange(batch["input_masked_imgs"], "b f c h w -> b c f h w").to(dtype=weight_dtype)  # b c f h w
                # mask video
                mask_values = rearrange(batch["input_masks"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                
                ref_image = rearrange(batch["ref_image"], "b c h w -> b c h w").to(dtype=weight_dtype)
                
                latent_masks = rearrange(F.interpolate(mask_values, scale_factor=1/16), "(b f) c h w -> b c f h w", f=cfg.data.nframes)
                first_frame_mask = latent_masks[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=4)
                mask_lat_size = torch.concat([first_frame_mask, latent_masks[:, :, 1:, :]], dim=2)
                batch_size, _, _, latent_height, latent_width = mask_lat_size.shape
                mask_lat_size = mask_lat_size.view(batch_size, -1, 4, latent_height, latent_width)  # b, 21, 4, h, w
                mask_lat_size = mask_lat_size.transpose(1, 2)  # b c f h w
                
                # Visualize data (first 3 steps only)
                if step < 3 and accelerator.is_main_process:
                    import torchvision
                    vis_dir = os.path.join(cfg.experiment.output_dirpath, "vis_data")
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # video: b c f h w -> save first sample, all frames as grid
                    vid_frames = video[0].permute(1, 0, 2, 3)  # f c h w
                    vid_frames = (vid_frames * 0.5 + 0.5).clamp(0, 1)
                    torchvision.utils.save_image(vid_frames, f"{vis_dir}/step{step}_video.png", nrow=9)
                    
                    # cond_values (masked video): b c f h w
                    cond_frames = cond_values[0].permute(1, 0, 2, 3)  # f c h w
                    cond_frames = (cond_frames * 0.5 + 0.5).clamp(0, 1)
                    torchvision.utils.save_image(cond_frames, f"{vis_dir}/step{step}_cond_masked.png", nrow=9)
                    
                    # mask_values: (b*f) c h w -> first batch
                    mask_frames = mask_values[:cfg.data.nframes]  # f c h w
                    torchvision.utils.save_image(mask_frames, f"{vis_dir}/step{step}_mask.png", nrow=9)
                    
                    # ref_image: b c h w
                    ref_vis = (ref_image[0:1] * 0.5 + 0.5).clamp(0, 1)
                    torchvision.utils.save_image(ref_vis, f"{vis_dir}/step{step}_ref_image.png")
                    
                    # mask_lat_size: b c f h w -> resize for visualization
                    mask_lat_vis = mask_lat_size[0, 0]  # f h w (first channel)
                    mask_lat_vis = mask_lat_vis.unsqueeze(1).repeat(1, 3, 1, 1)  # f 3 h w
                    mask_lat_vis = F.interpolate(mask_lat_vis, scale_factor=8, mode='nearest')
                    torchvision.utils.save_image(mask_lat_vis, f"{vis_dir}/step{step}_mask_latent.png", nrow=9)
                    
                    logger.info(f"Saved visualization for step {step} to {vis_dir}")
                
                with torch.no_grad():
                    latents = prepare_latents(vae, video, device=accelerator.device, dtype=weight_dtype)
                    cond_latents = prepare_latents(vae, cond_values, device=accelerator.device, dtype=weight_dtype)
                    
                    image_embeds = get_clip_image_embeds(
                        image_encoder, 
                        image_processor, 
                        ref_image, 
                        clip_proj,
                        device=accelerator.device, 
                        dtype=weight_dtype
                    )

                batch_size = latents.size(0)
                noise = torch.randn(latents.shape, device=accelerator.device, dtype=weight_dtype, generator=generator)
                timestep_id = torch.randint(0, noise_scheduler.num_train_timesteps, (1,))
                timestep = noise_scheduler.timesteps[timestep_id].to(dtype=weight_dtype, device=accelerator.device)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timestep)
                training_target = noise_scheduler.training_target(latents, noise, timestep)

                denoised_latents = transformer(
                    hidden_states=torch.concat([noisy_model_input, cond_latents, mask_lat_size], dim=1),  # concat along c-channel
                    timestep=timestep,
                    encoder_hidden_states=image_embeds, 
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

                weighting = noise_scheduler.training_weight(timestep)
                weight_mask = mask_lat_size[:,0:1]
                loss = torch.mean(
                    (weighting * (denoised_latents.float() - training_target.float()) ** 2 * (1 + weight_mask.float())).reshape(training_target.shape[0], -1),
                    1,
                )

                assert not torch.isnan(loss).any(), "NaN loss detected"

                accelerator.backward(loss)
                if cfg.hparams.gradient_precision == "fp32":
                    for param in all_trainable_parameters:
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(torch.float32)
                
                grad_norm = max_gradient(all_trainable_parameters)

                if accelerator.sync_gradients:
                    if accelerator.distributed_type == DistributedType.DEEPSPEED:
                        grad_norm = transformer.get_global_grad_norm()

                    elif cfg.hparams.max_grad_norm > 0:
                        if cfg.hparams.grad_clip_method == "norm":
                            grad_norm = accelerator.clip_grad_norm_(
                                all_trainable_parameters, cfg.hparams.max_grad_norm
                            )
                        elif cfg.hparams.grad_clip_method == "value":
                            grad_norm = accelerator.clip_grad_value_(
                                all_trainable_parameters, cfg.hparams.max_grad_norm
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
                    ema_model.step(parameters=all_trainable_parameters, global_step=global_step)
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

            if global_step >= cfg.hparams.max_train_steps:
                logger.info(f"max training steps={cfg.hparams.max_train_steps!r} reached.")
                break
        
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.hparams.ema.use_ema and ema_model is not None:
            ema_model.copy_to(all_trainable_parameters)

        transformer = unwrap_model(transformer)

        transformer.save_pretrained(
            output_dirpath
        )
    accelerator.wait_for_everyone()

    memory_statistics = get_memory_statistics()
    logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
