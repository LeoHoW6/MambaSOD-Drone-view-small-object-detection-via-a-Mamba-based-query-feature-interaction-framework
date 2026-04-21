"""
CrossMamba: a cross-attention variant of the Mamba selective state-space model.

Unlike the standard Mamba block where (B, C, dt) are all produced from the
same input sequence x, CrossMamba splits the roles:
  - the context sequence (hidden_states) produces (B, dt) and is scanned,
  - the query sequence  (query)         produces C,
so that the SSM state accumulated from the context is read out at positions
determined by the query. This gives a decoder-style cross interaction while
keeping the linear-time scan of Mamba.

Adapted from the Mamba repository (Gu & Dao, 2023):
  https://github.com/state-spaces/mamba
Licensed under Apache-2.0.

This module only supports full-sequence forward (the fast fused kernel).
Autoregressive single-token decoding is not implemented because MambaSOD
runs the decoder in a single shot per image.
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat

from selective_scan_interface_ca import mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


class Mamba(nn.Module):
    """CrossMamba mixer.

    Args:
        d_model:  model dimension (D)
        d_state:  SSM state dimension (N), larger = more history capacity
        d_conv:   kernel size of the depthwise conv before the scan
        expand:   inner-dim expansion factor (inner = expand * d_model)
        dt_rank:  rank of the delta (time-step) projection, 'auto' -> ceil(D/16)
        layer_idx: optional index, kept for API compatibility
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        # Context projection: produces (x, z) of shape (B, L, 2*d_inner).
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2,
                                 bias=bias, **factory_kwargs)
        # Query projection: produces the features used to derive C.
        self.in_proj_q = nn.Linear(self.d_model, self.d_inner,
                                   bias=bias, **factory_kwargs)

        # Depthwise 1D conv applied to x before the SSM scan.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        # Projects x -> (dt, B). C is produced from the query via q_proj.
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state,
            bias=False, **factory_kwargs
        )
        self.q_proj = nn.Linear(
            self.d_inner, self.d_state, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner,
                                 bias=True, **factory_kwargs)

        # dt projection init, preserves variance at init.
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias init: softplus(dt_bias) lies in [dt_min, dt_max].
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D-real initialization for the state matrix A.
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n", d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # Skip connection parameter D.
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model,
                                  bias=bias, **factory_kwargs)

    def forward(self, query, hidden_states, inference_params=None):
        """
        Args:
            query:         (B, L, D) query sequence
            hidden_states: (B, L, D) context sequence (same length as query)
        Returns:
            out: (B, L, D), same shape as hidden_states
        """
        assert inference_params is None, \
            "CrossMamba only supports full-sequence forward."
        assert causal_conv1d_fn is not None, \
            "causal_conv1d is required; install causal-conv1d."

        _, seqlen, _ = hidden_states.shape

        # Project context: (B, L, D) -> (B, 2*d_inner, L)
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # Project query: (B, L, D) -> (B, d_inner, L)
        qfc = rearrange(
            self.in_proj_q.weight @ rearrange(query, "b l d -> d (b l)"),
            "d (b l) -> b d l", l=seqlen,
        )
        if self.in_proj_q.bias is not None:
            qfc = qfc + rearrange(self.in_proj_q.bias.to(dtype=qfc.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        out = mamba_inner_fn(
            xz,
            qfc,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.q_proj.weight,
            self.dt_proj.weight,
            self.out_proj.weight,
            self.out_proj.bias,
            A,
            None,  # input-dependent B is produced inside the fused kernel
            None,  # input-dependent C is produced inside the fused kernel
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return out
