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
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from moviepy import ImageSequenceClip

from config import Config  # isort:skip
from optim import get_optimizer, max_gradient  # isort: skip
from dataloading.dataset_s1 import BaseDataset
from dataloading import utils
from models.transformer_wan import WanTransformer3DModel
from models.autoencoder_kl_wan import AutoencoderKLWan
from models.flow_match import FlowMatchScheduler
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from pipelines.pipeline_wan_inpainting import WanPipeline, retrieve_latents, prompt_clean
from ema import EMAModel # isort: skip
from utils_inference.video_writer import TensorSaveVideo



logging.basicConfig(
    #filename='train.log',
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
    force=True,
    handlers=[logging.StreamHandler()],
)
warnings.filterwarnings("ignore")  # ignore warning
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
            frame_gray = (frame_gray > 5).astype(np.uint8)*255
            frame_gray = frame_gray[None, :, :]
            frames.append(frame_gray)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    # Release the video capture object
    cap.release()

    return np.array(frames), fps

def log_validation(vae, tokenizer, text_encoder, transformer, cfg, accelerator, weigth_dtype, global_step):
    try:
        logger.info("Running validation... ")
        transformer3d_val = WanTransformer3DModel.from_pretrained(
            "/mnt/cfs/shanhai/wangsiyuan/wan_eraser/converted_ckpts/checkpoint-step00000001/transformer/"
        ).to(weigth_dtype)
        transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer).state_dict())
        #transformer3d_val = accelerator.unwrap_model(transformer)

        scheduler = FlowMatchScheduler(shift=7, sigma_min=0.0, extra_one_step=True)
        pipeline = WanPipeline(
            tokenizer=tokenizer,
            vae=accelerator.unwrap_model(vae).to(weigth_dtype),
            text_encoder=accelerator.unwrap_model(text_encoder),
            transformer=transformer3d_val,
            scheduler=scheduler,
        )
        pipeline = pipeline.to(accelerator.device)
        if cfg.experiment.random_seed is None:
            generator = None
        else:
            generator = torch.Generator(device=accelerator.device).manual_seed(cfg.experiment.random_seed)
        for idx, validation_prompt in enumerate(cfg.experiment.validation_prompts):
            video_path, mask_path, prompt, negative_prompt = validation_prompt
            
            # preprocess
            infer_len = 81
            orig_video, fps = read_video_cv2(video_path)
            orig_mask, _ = read_video_cv2(mask_path, mask=True)
            video_frame_len = len(orig_video) // infer_len * infer_len
            H, W, _ = orig_video[0].shape
            infer_h = 720
            infer_w = 1280
            generated_frames = []
            for start_idx in range(0, video_frame_len, infer_len):
                end_idx = start_idx + infer_len
                print("start_idx and end_idx: ", start_idx, end_idx)
                batch_orig_frames = orig_video[list(range(start_idx, end_idx))]  # [f h w c]
                batch_orig_mask = orig_mask[list(range(start_idx, end_idx))]  # [f c h w ]

                #mask_seq = rearrange(batch_orig_mask,"f c h w -> f c h w")
                mask_seq = torch.from_numpy(batch_orig_mask).to(accelerator.device)
                mask_seq = F.interpolate(mask_seq, size=(infer_h, infer_w), mode="nearest-exact")
                mask_seq = mask_seq.to(torch.float16) / 255.0
                #print(mask_seq.shape)
                #mask_seq = mask_seq / 255.0
                first_frame_mask = mask_seq[0:1, :, :]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=0, repeats=4)
                mask_lat_size = torch.concat([first_frame_mask, mask_seq[1:, :, :, :]], dim=0)  # f c h w
                mask_lat_size = F.interpolate(mask_lat_size, scale_factor=1/16, mode="nearest-exact")
                #print(mask_lat_size.shape)
                num_frames, _, latent_height, latent_width = mask_lat_size.shape
                mask_lat_size = mask_lat_size.view(1, num_frames//4, 4, latent_height, latent_width)  # b f c h w
                mask_lat_size = mask_lat_size.transpose(1, 2)                                         # b c f h w

                # prepare condition
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=weigth_dtype):
                        video_transforms = transforms.Compose(
                            [
                                transforms.Lambda(lambda x: x / 255.0),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                            ]
                        )
                        cond_video = torch.from_numpy(np.stack(batch_orig_frames, axis=0)).permute(0, 3, 1, 2)  # F C H W
                        cond_video = F.interpolate(cond_video, size=(infer_h, infer_w), mode="bicubic")
                        cond_video = torch.stack([video_transforms(x) for x in cond_video], dim=0)
                        with torch.inference_mode():
                            image_or_video = cond_video.to(device=accelerator.device, dtype=weigth_dtype)
                            #image_or_video = (image_or_video * (1 - mask_seq)).unsqueeze(0)
                            image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous() # [B, F, C, H, W] -> [B, C, F, H, W]
                            cond_latents = prepare_latents(vae, image_or_video)
                            cond_latents = cond_latents.to(dtype=weigth_dtype)
                        video = pipeline(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            height=infer_h,
                            width=infer_w,
                            num_frames=81,
                            num_inference_steps=50,
                            guidance_scale=3.0,  # 3.0
                            generator=generator,
                            cond_latents=cond_latents,
                            cond_masks=mask_lat_size,
                        ).frames
                        video_frames = (video * 255.0).astype(np.uint8)[0]
                        generated_frames += [cv2.resize(video_frame, (W, H)) for video_frame in video_frames]
            output_path = f"{cfg.experiment.output_dirpath}/samples/sample-{global_step}/{idx}.mp4"
            save_video_with_numpy(generated_frames, output_path, fps)
        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error with info {e}")
        return None


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
        logger.warning("No CUDA, MPS, or ROCm device found. Memory statistics are not available.")

    return {
        "memory_allocated": round(bytes_to_gigabytes(memory_allocated), ndigits=precision),
        "memory_reserved": round(bytes_to_gigabytes(memory_reserved), ndigits=precision),
        "max_memory_allocated": round(bytes_to_gigabytes(max_memory_allocated), ndigits=precision),
        "max_memory_reserved": round(bytes_to_gigabytes(max_memory_reserved), ndigits=precision),
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
    #latents = retrieve_latents(vae.encode(image_or_video), generator, sample_mode="argmax")
    latents = retrieve_latents(vae.encode(image_or_video), generator, sample_mode="sample")
    #latents = retrieve_latents(vae.encode(image_or_video), latents_mean, latents_std, generator, sample_mode="argmax")
    latents = (latents - latents_mean) * latents_std

    latents = latents.to(dtype=dtype) 
    return latents


def get_nb_trainable_parameters(mod: torch.nn.Module):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    # note: same as PeftModel.get_nb_trainable_parameters
    trainable_params = 0
    all_param = 0
    for _, param in mod.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
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

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds


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
    tensor_writer = TensorSaveVideo()


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
    #for t2v use text encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=load_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    # for i2v use image encoder
    '''image_encoder = CLIPVisionModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="image_encoder",
        torch_dtype=load_dtype,
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="image_processor"
    )'''
    logger.info(f"Load transformer model from {cfg.model.pretrained_model_name_or_path!r}")

    # ignore_mismatched_sizes=True will init the patch_embedding weght
    transformer = WanTransformer3DModel.from_pretrained(
        "/mnt/cfs/shanhai/wangsiyuan/wan_2/wan_eraser/converted_ckpts/checkpoint-step00000001",
        #"/mnt/cfs/shanhai/liuh/Wan2.2-TI2V-5B-Diffusers/",
        subfolder="transformer",
        torch_dtype=load_dtype,
    )
    logger.info(f"Loaded transformer model from {cfg.model.pretrained_model_name_or_path!r}")
    '''
    with torch.no_grad():
        logger.info("expand transformer input channels")
        initial_input_channels = transformer.config.in_channels

        
        new_img_in = nn.Conv3d(transformer.config.in_channels * 2 + 4, 
                               transformer.config.num_attention_heads * transformer.config.attention_head_dim,
                               kernel_size=transformer.config.patch_size, 
                               stride=transformer.config.patch_size)
        
        #conv3d.weight = (out_channels, in_channels, kernel_depth, kernel_height, kernel_width)
        new_img_in.weight.zero_()
        new_img_in.weight[:, :initial_input_channels, ...].copy_(transformer.patch_embedding.weight)
        if transformer.patch_embedding.bias is not None:
            new_img_in.bias.copy_(transformer.patch_embedding.bias)
        transformer.patch_embedding = new_img_in
        assert torch.all(transformer.patch_embedding.weight[:, initial_input_channels:, ...] == 0)
        transformer.register_to_config(in_channels=initial_input_channels * 2 + 4, out_channels=initial_input_channels)
        logger.info(f"expanded transformer patch_embedding input channels")
    '''


    accelerator.wait_for_everyone()
    

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
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
    # image_encoder.requires_grad_(False)
    # transformer.requires_grad_(False)

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info(f"dit dtype: {next(transformer.parameters()).dtype!r}")

    trainable_params, all_param = get_nb_trainable_parameters(transformer)
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
    
    # for name, param in transformer.named_parameters():
    #     if any(k in name for k in ['add', 'patch_embedding', 'audio_proj']):
    #         param.requires_grad_(True)


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

    # check parameters
    '''
    if accelerator.is_main_process:
        rec_txt1 = open('rec_para.txt', 'w')
        rec_txt2 = open('rec_para_train.txt', 'w')
        for name, para in transformer.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()
    '''
    # ======================================================
    # 3. build dataset and dataloaders
    # ======================================================
    train_dataset = BaseDataset(cfg.data)
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
            
            # we'll temporarily overwrite the LoRA parameters with the EMA parameters to save it.
            logger.info("Saving EMA model to disk.")
            trainable_parameters = [p for p in primary_model.parameters() if p.requires_grad]
            ema_model.store(trainable_parameters)
            ema_model.copy_to(trainable_parameters)
           
            ema_model.save_pretrained(os.path.join(output_dir, "ema"))
            ema_model.restore(trainable_parameters)
        
        
        for model in models:
            if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                model = unwrap_model(model)
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

                transformer_ = WanTransformer3DModel.from_pretrained(
                    #cfg.model.pretrained_model_name_or_path,
                    '/mnt/cfs/shanhai/wangsiyuan/wan_2/wan_eraser/converted_ckpts/checkpoint-step00000001"',
                    subfolder="transformer"
                )
                transformer_.to(accelerator.device, weight_dtype)
                init_under_meta = True


        load_transformer_model = WanTransformer3DModel.from_pretrained(
            input_dir, subfolder="transformer"
        )
        transformer_.register_to_config(**load_transformer_model.config)
        transformer_.load_state_dict(load_transformer_model.state_dict(), assign=init_under_meta)
        del load_transformer_model

        if cfg.hparams.mixed_precision == "fp16":
            cast_training_params([transformer_])

        logger.info(f"Completed loading checkpoint from Path: {input_dir!r}")


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # The epoch_size attribute of StreamingDataset is the number of samples per epoch of training.
    # The __len__() method returns the epoch_size divided by the number of devices – it is the number of samples seen per device, per epoch.
    # The size() method returns the number of unique samples in the underlying dataset.
    # Due to upsampling/downsampling, size() may not be the same as epoch_size.

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
    
    transformer, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(transformer, train_dataloader, optimizer, lr_scheduler)


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
    num_trainable_parameters = sum(p.numel() for p in transformer_parameters)

    # fmt: off
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
            # Get the mos recent checkpoint
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

    # model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
    generator = torch.Generator(device=accelerator.device)
    # scheduler_sigmas = noise_scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
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
                #print("######cond_values shape#######")
                # mask video
                mask_values = rearrange(batch["input_masks"], "b f c h w -> (b f) c h w").to(dtype=weight_dtype)
                #print(mask_values.shape)
                ####write tensor data######
                '''video_tensor = rearrange(video, "b c f h w -> b f h w c")
                video_tensor = rearrange(video_tensor, "b f h w c -> (b f) h w c")
                video_tensor_path = 'data/video_input_{}.mp4'.format(step)
                tensor_writer.torch_nhwc_to_video(video_tensor, video_tensor_path, 25)
                cond_values_tensor = rearrange(cond_values, "b c f h w -> b f h w c")
                cond_values_tensor = rearrange(cond_values_tensor, "b f h w c -> (b f) h w c")
                cond_values_tensor_path = 'data/video_mask_input_{}.mp4'.format(step)
                tensor_writer.torch_nhwc_to_video(cond_values_tensor.float(), cond_values_tensor_path, 25)
                print(cond_values.shape)
                mask_values_path = 'data/mask_input_{}.mp4'.format(step)
                tensor_writer.torch_nchw_to_video(mask_values.float(), mask_values_path, 25)'''
                ####################
                latent_masks = rearrange(F.interpolate(mask_values, scale_factor=1/16), "(b f) c h w -> b c f h w", f=cfg.data.nframes)
                first_frame_mask = latent_masks[:, :, 0:1]
                first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=4)
                mask_lat_size = torch.concat([first_frame_mask, latent_masks[:, :, 1:, :]], dim=2)
                batch_size, _, _, latent_height, latent_width = mask_lat_size.shape
                mask_lat_size = mask_lat_size.view(batch_size, -1, 4, latent_height, latent_width)  # b, 21, 4, h, w
                mask_lat_size = mask_lat_size.transpose(1, 2)  # b c f h w
                # using vae encoder to encode input to latent input
                prompt = batch["caption"]
                with torch.no_grad():
                    latents = prepare_latents(vae, video, device=accelerator.device, dtype=weight_dtype)
                    cond_latents = prepare_latents(vae, cond_values, device=accelerator.device, dtype=weight_dtype)
                    prompt_embeds = get_t5_prompt_embeds(text_encoder, tokenizer, prompt, device=latents.device, dtype=weight_dtype)
                batch_size = latents.size(0)
                noise = torch.randn(latents.shape, device=accelerator.device, dtype=weight_dtype, generator=generator)
                timestep_id = torch.randint(0, noise_scheduler.num_train_timesteps, (1,))
                timestep = noise_scheduler.timesteps[timestep_id].to(dtype=weight_dtype, device=accelerator.device)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timestep)
                training_target = noise_scheduler.training_target(latents, noise, timestep)

                denoised_latents = transformer(
                    hidden_states=torch.concat([noisy_model_input, cond_latents, mask_lat_size], dim=1),  # concat along c-channel
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

                #from jj
                weighting = noise_scheduler.training_weight(timestep)
                weight_mask = mask_lat_size[:,0:1]
                loss = torch.mean(
                    (weighting * (denoised_latents.float() - training_target.float()) ** 2 * (1 + weight_mask.float())).reshape(training_target.shape[0], -1),
                    1,
                )

                #loss = torch.nn.functional.mse_loss(denoised_latents.float(), training_target.float())
                #loss = loss * noise_scheduler.training_weight(timestep)
                
                assert torch.isnan(loss) == False, "NaN loss detected"

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
                    
                    if global_step % cfg.experiment.validation_steps == 0:
                        log_validation(
                            vae, 
                            tokenizer,
                            text_encoder, 
                            transformer, 
                            cfg, 
                            accelerator, 
                            weight_dtype, 
                            global_step
                        )
                    
                    

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
            ema_model.copy_to(transformer_parameters)

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
                
