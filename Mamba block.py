"""
Mamba block wrappers used in MambaSOD.

Block:       wraps a Mamba mixer with RMSNorm / LayerNorm and a residual
             connection (pre-norm style).
CrossBlock:  wraps a CrossMamba mixer. The query sequence and the
             context (hidden_states) sequence are normalized independently
             before being fed into the mixer.

Adapted from the Mamba repository (Gu & Dao, 2023):
  https://github.com/state-spaces/mamba
Licensed under Apache-2.0.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None


class CrossBlock(nn.Module):
    """Pre-norm block for CrossMamba: normalizes query and context separately."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False):
        super().__init__()
        assert not fused_add_norm, \
            "fused_add_norm is not supported; CrossBlock uses the slow path."
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.norm_query = norm_cls(dim)

        if RMSNorm is not None:
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported."

    def forward(self, query, hidden_states: Tensor,
                residual: Optional[Tensor] = None, inference_params=None):
        """
        Args:
            query:         (B, Lq, D) query sequence
            hidden_states: (B, Lk, D) context sequence
            residual:      (B, Lk, D) or None, added to hidden_states before norm
        Returns:
            (out, residual): out of the cross mixer and the pre-norm residual
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        query = self.norm_query(query.to(dtype=self.norm_query.weight.dtype))
        hidden_states = self.mixer(query, hidden_states, inference_params=inference_params)
        return hidden_states, residual


class Block(nn.Module):
    """Pre-norm block for self-Mamba."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False):
        super().__init__()
        assert not fused_add_norm, \
            "fused_add_norm is not supported; Block uses the slow path."
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

        if RMSNorm is not None:
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), \
                "Only LayerNorm and RMSNorm are supported."

    def forward(self, hidden_states: Tensor,
                residual: Optional[Tensor] = None,
                inference_params=None, **mixer_kwargs):
        """
        Args:
            hidden_states: (B, L, D)
            residual:      (B, L, D) or None
        Returns:
            (out, residual)
        """
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        hidden_states = self.mixer(hidden_states, inference_params=inference_params,
                                   **mixer_kwargs)
        return hidden_states, residual