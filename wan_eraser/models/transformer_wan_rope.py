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

"""
Modified transformer with fixed reference frame RoPE position at index 60.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class WanRotaryPosEmbedFixedRef(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        fixed_ref_position: int = 60,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.fixed_ref_position = fixed_ref_position

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
        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def _compute_freqs_for_segment(
        self,
        num_frames: int,
        height: int,
        width: int,
        frame_start_idx: int = 0,
        use_fixed_ref_position: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos_split = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin_split = self.freqs_sin.split(split_sizes, dim=1)

        if use_fixed_ref_position:
            ref_idx = self.fixed_ref_position
            freqs_cos_f = freqs_cos_split[0][ref_idx:ref_idx+1].view(1, 1, 1, -1).expand(num_frames, height, width, -1)
            freqs_sin_f = freqs_sin_split[0][ref_idx:ref_idx+1].view(1, 1, 1, -1).expand(num_frames, height, width, -1)
        else:
            freqs_cos_f = freqs_cos_split[0][frame_start_idx:frame_start_idx + num_frames]
            freqs_cos_f = freqs_cos_f.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1)
            freqs_sin_f = freqs_sin_split[0][frame_start_idx:frame_start_idx + num_frames]
            freqs_sin_f = freqs_sin_f.view(num_frames, 1, 1, -1).expand(num_frames, height, width, -1)

        freqs_cos_h = freqs_cos_split[1][:height].view(1, height, 1, -1).expand(num_frames, height, width, -1)
        freqs_cos_w = freqs_cos_split[2][:width].view(1, 1, width, -1).expand(num_frames, height, width, -1)
        freqs_sin_h = freqs_sin_split[1][:height].view(1, height, 1, -1).expand(num_frames, height, width, -1)
        freqs_sin_w = freqs_sin_split[2][:width].view(1, 1, width, -1).expand(num_frames, height, width, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1)
        
        return freqs_cos, freqs_sin

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_segments: Optional[List[Tuple[int, bool]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        if frame_segments is None:
            freqs_cos, freqs_sin = self._compute_freqs_for_segment(
                ppf, pph, ppw, frame_start_idx=0, use_fixed_ref_position=False
            )
            freqs_cos = freqs_cos.reshape(1, ppf * pph * ppw, 1, -1)
            freqs_sin = freqs_sin.reshape(1, ppf * pph * ppw, 1, -1)
            return freqs_cos, freqs_sin

        all_freqs_cos = []
        all_freqs_sin = []
        current_frame_idx = 0
        
        for segment_frames, is_reference in frame_segments:
            pp_segment_frames = segment_frames // p_t
            
            freqs_cos, freqs_sin = self._compute_freqs_for_segment(
                pp_segment_frames, pph, ppw,
                frame_start_idx=current_frame_idx,
                use_fixed_ref_position=is_reference,
            )
            freqs_cos = freqs_cos.reshape(pp_segment_frames * pph * ppw, -1)
            freqs_sin = freqs_sin.reshape(pp_segment_frames * pph * ppw, -1)
            all_freqs_cos.append(freqs_cos)
            all_freqs_sin.append(freqs_sin)
            
            if not is_reference:
                current_frame_idx += pp_segment_frames

        freqs_cos = torch.cat(all_freqs_cos, dim=0).unsqueeze(0).unsqueeze(2)
        freqs_sin = torch.cat(all_freqs_sin, dim=0).unsqueeze(0).unsqueeze(2)

        return freqs_cos, freqs_sin


class WanTransformer3DModelFixedRef(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
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
        use_fixed_ref_rope: bool = True,
        fixed_ref_position: int = 60,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        if use_fixed_ref_rope:
            self.rope = WanRotaryPosEmbedFixedRef(
                attention_head_dim, patch_size, rope_max_seq_len,
                fixed_ref_position=fixed_ref_position
            )
        else:
            self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

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
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        frame_segments: Optional[List[Tuple[int, bool]]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if self.config.use_fixed_ref_rope and frame_segments is not None:
            rotary_emb = self.rope(hidden_states, frame_segments=frame_segments)
        else:
            rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

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

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

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

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
