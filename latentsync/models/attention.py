# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.attention import FeedForward, AdaLayerNorm

from einops import rearrange, repeat


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        add_audio_layer=False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    add_audio_layer=add_audio_layer,
                )
                for d in range(num_layers)
            ]
        )

        # Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        upcast_attention: bool = False,
        add_audio_layer=False,
    ):
        super().__init__()
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.add_audio_layer = add_audio_layer

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

        # Cross-attn
        if add_audio_layer:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None
    ):
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.attn2 is not None and encoder_hidden_states is not None:
            if encoder_hidden_states.dim() == 4:
                encoder_hidden_states = rearrange(encoder_hidden_states, "b f s d -> (b f) s d")
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states


MPS_ATTENTION_BUDGET_BYTES = int(float(os.environ.get("MPS_ATTENTION_BUDGET_MB", "1024")) * (1 << 20))


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        norm_num_groups: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax

        self.scale = dim_head**-0.5

        self.heads = heads

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=inner_dim, num_groups=norm_num_groups, eps=1e-5, affine=True)
        else:
            self.group_norm = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def split_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    def concat_heads(self, tensor):
        batch_size, heads, seq_len, head_dim = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = tensor.reshape(batch_size, seq_len, heads * head_dim)
        return tensor

    @staticmethod
    def scaled_dot_product_attention(query, key, value, attention_mask=None):
        # Use PyTorch native implementation of FlashAttention-2
        if query.device.type != "mps":
            return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        # MPS materializes the full attention matrix, which can exceed the 16 GiB Metal buffer limit at 512x512
        # (32 frames x 8 heads x 4096 tokens). Process the batch in chunks; the result is identical.
        batch_size, heads, q_len, _ = query.shape
        score_bytes_per_sample = heads * q_len * key.shape[2] * query.element_size()
        chunk_size = max(1, MPS_ATTENTION_BUDGET_BYTES // score_bytes_per_sample)
        if chunk_size >= batch_size:
            return F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)

        outputs = []
        for i in range(0, batch_size, chunk_size):
            mask = attention_mask
            if mask is not None and mask.ndim == 4 and mask.shape[0] == batch_size:
                mask = mask[i : i + chunk_size]
            outputs.append(
                F.scaled_dot_product_attention(
                    query[i : i + chunk_size], key[i : i + chunk_size], value[i : i + chunk_size], attn_mask=mask
                )
            )
        return torch.cat(outputs, dim=0)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        query = self.split_heads(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        key = self.split_heads(key)
        value = self.split_heads(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        hidden_states = self.scaled_dot_product_attention(query, key, value, attention_mask)

        hidden_states = self.concat_heads(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
