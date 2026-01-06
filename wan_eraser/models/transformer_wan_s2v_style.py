# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

from models.transformer_wan import (
    WanAttnProcessor,
    WanAttention,
    WanImageEmbedding,
    WanTimeTextImageEmbedding,
    WanRotaryPosEmbed,
    WanTransformerBlock,
)


logger = logging.get_logger(__name__)


@amp.autocast(enabled=False)
def rope_apply(x, freqs):
    """Apply rotary position embedding to input tensor."""
    # x: [B, S, N, D]
    # freqs: [B, S, 1, D//2] complex
    b, s, n, d = x.shape
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(b, s, n, -1, 2))
    x_rot = torch.view_as_real(x_complex * freqs).flatten(3)
    return x_rot.float()


def rope_precompute_s2v_style(
    hidden_states: torch.Tensor,
    grid_sizes: List[List[torch.Tensor]],
    freqs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    Precompute RoPE frequencies for S2V-style model with separate video and ref segments.
    
    Args:
        hidden_states: [B, S, D] flattened hidden states
        grid_sizes: List of [start_grid, end_grid, range_grid] for each segment
        freqs: Tuple of (freq_t, freq_h, freq_w) precomputed frequencies
        num_heads: number of attention heads
        head_dim: dimension per head
    
    Returns:
        freqs_complex: [B, S, 1, D//2] complex frequencies
    """
    b, s = hidden_states.shape[:2]
    c = head_dim // 2
    
    # Split frequencies
    freq_t, freq_h, freq_w = freqs
    
    output = torch.zeros(b, s, 1, c, dtype=torch.complex128, device=hidden_states.device)
    
    seq_offset = 0
    for grid_info in grid_sizes:
        start_grid, end_grid, range_grid = grid_info
        
        for i in range(b):
            f_start, h_start, w_start = start_grid[i]
            f_end, h_end, w_end = end_grid[i]
            t_f, t_h, t_w = range_grid[i]
            
            seq_f = int(f_end - f_start)
            seq_h = int(h_end - h_start)
            seq_w = int(w_end - w_start)
            seq_len = seq_f * seq_h * seq_w
            
            if seq_len == 0:
                continue
            
            # Sample frequency indices
            import numpy as np
            if t_f > 0:
                if f_start >= 0:
                    f_indices = np.linspace(f_start.item(), (t_f + f_start).item() - 1, seq_f).astype(int).tolist()
                else:
                    f_indices = np.linspace(-f_start.item(), (-t_f - f_start).item() + 1, seq_f).astype(int).tolist()
                
                h_indices = np.linspace(h_start.item(), (t_h + h_start).item() - 1, seq_h).astype(int).tolist()
                w_indices = np.linspace(w_start.item(), (t_w + w_start).item() - 1, seq_w).astype(int).tolist()
                
                # Get frequencies
                freqs_f = freq_t[f_indices] if f_start >= 0 else freq_t[f_indices].conj()
                freqs_f = freqs_f.view(seq_f, 1, 1, -1)
                
                freqs_combined = torch.cat([
                    freqs_f.expand(seq_f, seq_h, seq_w, -1),
                    freq_h[h_indices].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                    freq_w[w_indices].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                ], dim=-1).reshape(seq_len, 1, -1)
                
                output[i, seq_offset:seq_offset + seq_len] = freqs_combined
        
        seq_offset += seq_len
    
    return output


class WanRotaryPosEmbedS2VStyle(nn.Module):
    """
    S2V-style RoPE with fixed reference position at index 30.
    Reference frames use a fixed temporal position while video frames use sequential positions.
    """
    
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        fixed_ref_position: int = 30,
    ):
        super().__init__()
        
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.fixed_ref_position = fixed_ref_position
        
        # Compute dimension splits like S2V
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self.dims = [t_dim, h_dim, w_dim]
        self.freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        
        freqs_cos = []
        freqs_sin = []
        for dim in self.dims:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta,
                use_real=True, repeat_interleave_real=True, freqs_dtype=self.freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)
        
        self.register_buffer("freq_t_cos", freqs_cos[0], persistent=False)
        self.register_buffer("freq_h_cos", freqs_cos[1], persistent=False)
        self.register_buffer("freq_w_cos", freqs_cos[2], persistent=False)
        self.register_buffer("freq_t_sin", freqs_sin[0], persistent=False)
        self.register_buffer("freq_h_sin", freqs_sin[1], persistent=False)
        self.register_buffer("freq_w_sin", freqs_sin[2], persistent=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_video_frames: int,
        num_ref_frames: int = 1,
        height: int = None,
        width: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE frequencies for video + reference frames.
        
        Args:
            hidden_states: [B, C, F_total, H, W] input tensor
            num_video_frames: number of video frames (excluding ref)
            num_ref_frames: number of reference frames (default 1)
            height: height in patches (if None, computed from hidden_states)
            width: width in patches (if None, computed from hidden_states)
        
        Returns:
            freqs_cos, freqs_sin: RoPE frequencies
        """
        batch_size, num_channels, num_frames, H, W = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        
        if height is None:
            height = H // p_h
        if width is None:
            width = W // p_w
        
        # Video frames: sequential from 0
        video_pp_frames = num_video_frames // p_t
        ref_pp_frames = num_ref_frames // p_t
        
        # Split frequencies by dimension
        c = self.attention_head_dim // 2
        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]
        
        # Build video frequencies
        freqs_cos_list = []
        freqs_sin_list = []
        
        # 1. Video frames: sequential temporal positions 0 to video_pp_frames-1
        for f_idx in range(video_pp_frames):
            for h_idx in range(height):
                for w_idx in range(width):
                    freq_cos = torch.cat([
                        self.freq_t_cos[f_idx:f_idx+1],
                        self.freq_h_cos[h_idx:h_idx+1],
                        self.freq_w_cos[w_idx:w_idx+1]
                    ], dim=-1)
                    freq_sin = torch.cat([
                        self.freq_t_sin[f_idx:f_idx+1],
                        self.freq_h_sin[h_idx:h_idx+1],
                        self.freq_w_sin[w_idx:w_idx+1]
                    ], dim=-1)
                    freqs_cos_list.append(freq_cos)
                    freqs_sin_list.append(freq_sin)
        
        # 2. Reference frames: fixed temporal position at self.fixed_ref_position
        for f_idx in range(ref_pp_frames):
            for h_idx in range(height):
                for w_idx in range(width):
                    freq_cos = torch.cat([
                        self.freq_t_cos[self.fixed_ref_position:self.fixed_ref_position+1],
                        self.freq_h_cos[h_idx:h_idx+1],
                        self.freq_w_cos[w_idx:w_idx+1]
                    ], dim=-1)
                    freq_sin = torch.cat([
                        self.freq_t_sin[self.fixed_ref_position:self.fixed_ref_position+1],
                        self.freq_h_sin[h_idx:h_idx+1],
                        self.freq_w_sin[w_idx:w_idx+1]
                    ], dim=-1)
                    freqs_cos_list.append(freq_cos)
                    freqs_sin_list.append(freq_sin)
        
        # Stack all frequencies: [S, D]
        freqs_cos = torch.cat(freqs_cos_list, dim=0)  # [S, D]
        freqs_sin = torch.cat(freqs_sin_list, dim=0)  # [S, D]
        
        # Add batch and head dims: [1, S, 1, D]
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)
        
        return freqs_cos, freqs_sin


class WanTransformer3DModelS2VStyle(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    """
    S2V-style transformer for video inpainting.
    
    Key differences from standard WanTransformer3DModel:
    - Reference frame concatenated on temporal dimension (d), not channel
    - patch_embedding: 48 channels (model_input on channel: noisy + masked_video + mask)
    - Reference RoPE position fixed at 30
    - Trainable condition mask embedding for video/ref tokens
    """
    
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    # _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 100,  
        out_channels: int = 48, 
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        fixed_ref_position: int = 30,
    ) -> None:
        super().__init__()
        
        inner_dim = num_attention_heads * attention_head_dim
        self.out_channels = out_channels or in_channels
        self.inner_dim = inner_dim
        self.num_heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.fixed_ref_position = fixed_ref_position
        
        # S2V-style RoPE
        self.rope = WanRotaryPosEmbedS2VStyle(
            attention_head_dim, patch_size, rope_max_seq_len,
            fixed_ref_position=fixed_ref_position
        )
        
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.ref_patch_embedding = nn.Conv3d(48, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.trainable_cond_mask = nn.Embedding(2, inner_dim)
        
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )
        
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        ref_latents: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for S2V-style inpainting transformer.
        
        Args:
            hidden_states: [B, 48, F, H, W] model input (noisy + masked_video + mask on channel)
            ref_latents: [B, 16, 1, H, W] reference latents (single frame)
            timestep: [B] or [B, T] timesteps
            encoder_hidden_states: [B, L, D] text embeddings
            encoder_hidden_states_image: optional image embeddings
            return_dict: whether to return dict
            attention_kwargs: additional attention arguments
        
        Returns:
            output: [B, 16, F, H, W] denoised output (only video frames, not ref)
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
        
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
        
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        ref_batch, ref_ch, ref_frames, ref_h, ref_w = ref_latents.shape
        
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        ref_post_patch_frames = ref_frames // p_t
        
        # Compute RoPE for video + reference
        # Create dummy full tensor for RoPE computation
        full_frames = num_frames + ref_frames
        dummy_tensor = torch.zeros(batch_size, 1, full_frames, height, width, device=hidden_states.device)
        rotary_emb = self.rope(
            dummy_tensor,
            num_video_frames=num_frames,
            num_ref_frames=ref_frames,
            height=post_patch_height,
            width=post_patch_width,
        )
        
        # Patch embedding for video input (100 channels)
        video_hidden = self.patch_embedding(hidden_states)  # [B, D, F', H', W']
        video_hidden = video_hidden.flatten(2).transpose(1, 2)  # [B, S_video, D]
        video_seq_len = video_hidden.shape[1]
        
        # Patch embedding for reference (48 channels)
        ref_hidden = self.ref_patch_embedding(ref_latents)  # [B, D, F'_ref, H', W']
        ref_hidden = ref_hidden.flatten(2).transpose(1, 2)  # [B, S_ref, D]
        ref_seq_len = ref_hidden.shape[1]
        
        # Concatenate video + ref on sequence dimension (S2V style)
        hidden_states = torch.cat([video_hidden, ref_hidden], dim=1)  # [B, S_video + S_ref, D]
        
        # Create condition mask: 0 for video, 1 for reference
        cond_mask = torch.zeros(batch_size, video_seq_len + ref_seq_len, dtype=torch.long, device=hidden_states.device)
        cond_mask[:, video_seq_len:] = 1
        
        # Add trainable condition embedding
        hidden_states = hidden_states + self.trainable_cond_mask(cond_mask)
        
        # Process timestep
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None
        
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)
        
        # Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        
        # Output projection
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        
        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        
        # Extract only video tokens (not reference)
        video_output = hidden_states[:, :video_seq_len, :]  # [B, S_video, out_ch * prod(patch)]
        
        # Reshape back to video format
        video_output = video_output.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        video_output = video_output.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = video_output.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        
        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        
        if not return_dict:
            return (output,)
        
        return Transformer2DModelOutput(sample=output)
