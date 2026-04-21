"""
Cross-Mamba selective scan CUDA interface.

Provides `mamba_inner_fn`, a fused forward/backward kernel wrapper used by
CrossMamba (see cross_mamba.py). The difference from the standard
Mamba inner function is that the SSM output matrix C is produced from the
query projection `qfc` instead of from the context projection `x`, giving
the block a cross-attention-like behavior while preserving linear-time
scan.

Adapted from the Mamba repository (Gu & Dao, 2023):
  https://github.com/state-spaces/mamba
Licensed under Apache-2.0.

Requires:
  - mamba-ssm (provides `selective_scan_cuda`)
  - causal-conv1d (provides `causal_conv1d_cuda`)
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd

from einops import rearrange

try:
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_cuda = None

import selective_scan_cuda


class MambaInnerFn(torch.autograd.Function):
    """
    Fused cross-Mamba inner function.

    Inputs:
        xz:  (B, 2*d_inner, L)  context projection (splits into x, z)
        qfc: (B, d_inner, L)    query projection, used to produce C
        A:   (d_inner, d_state) state matrix
        ... plus learnable projection weights/biases.

    Returns:
        (B, L, d_model) output, already passed through out_proj.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, qfc,
                conv1d_weight, conv1d_bias,
                x_proj_weight, q_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None,
                delta_bias=None, B_proj_bias=None, C_proj_bias=None,
                delta_softplus=True, checkpoint_lvl=1):
        assert causal_conv1d_cuda is not None, \
            "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        assert B is None and C is None, \
            "CrossMamba always produces B and C internally from x_dbl / q_dbl."

        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        # Cast projection weights to autocast dtype when AMP is enabled.
        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            q_proj_weight = q_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)

        if xz.stride(-1) != 1:
            xz = xz.contiguous()

        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None

        # Depthwise causal conv (signature: x, w, bias, seq_idx, silu_activation).
        conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
            x, conv1d_weight, conv1d_bias, None, True
        )

        # (dt, B) from context; C comes from query.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)
        q_dbl = F.linear(rearrange(qfc, 'b d l -> (b l) d'), q_proj_weight)
        x_dbl = torch.cat([x_dbl, q_dbl], axis=-1)

        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                          "d (b l) -> b d l", l=L)

        ctx.is_variable_B = True
        ctx.is_variable_C = True
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None

        B = x_dbl[:, delta_rank:delta_rank + d_state]
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)",
                          l=L, two=2).contiguous()

        C = x_dbl[:, -d_state:]
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)",
                          l=L, two=2).contiguous()

        if D is not None:
            D = D.contiguous()

        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus
        )

        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.has_conv1d_bias = conv1d_bias is not None

        # Activation checkpointing: drop large intermediates and recompute them in backward.
        if checkpoint_lvl >= 1:
            conv1d_out, delta = None, None

        ctx.save_for_backward(xz, qfc, conv1d_weight, conv1d_bias, x_dbl,
                              x_proj_weight, q_proj_weight, delta_proj_weight,
                              out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, out)

        return F.linear(rearrange(out_z, "b d l -> b l d"),
                        out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        assert causal_conv1d_cuda is not None, \
            "causal_conv1d_cuda is not available."

        (xz, qfc, conv1d_weight, conv1d_bias, x_dbl,
         x_proj_weight, q_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias,
         scan_intermediates, out) = ctx.saved_tensors

        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        # Recompute dropped intermediates.
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
                x, conv1d_weight, conv1d_bias, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l=L)

        dxz = torch.empty_like(xz)
        dx, dz = dxz.chunk(2, dim=1)

        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)

        (dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz,
         out_z) = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C, D, z, delta_bias,
            dout_y, scan_intermediates, out, dz,
            ctx.delta_softplus, True
        )

        dout_proj_weight = torch.einsum(
            "eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)")
        )
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None

        dx_dbl = torch.empty_like(x_dbl)

        # Gradient w.r.t. B (from context branch).
        dB_proj_bias = None
        if not A.is_complex():
            dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
        else:
            dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)",
                           two=2).contiguous()
        dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
        dx_dbl[:, delta_rank:delta_rank + d_state] = dB

        # Gradient w.r.t. C (from query branch).
        dC_proj_bias = None
        if not A.is_complex():
            dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
        else:
            dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)",
                           two=2).contiguous()
        dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
        dx_dbl[:, -d_state:] = dC

        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)

        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum(
            "Br,Bd->rd",
            dx_dbl[:, :delta_rank + d_state],
            rearrange(conv1d_out, "b d l -> (b l) d")
        )
        dq_proj_weight = torch.einsum(
            "Br,Bd->rd",
            dx_dbl[:, -d_state:],
            rearrange(qfc, "b d l -> (b l) d")
        )

        dconv1d_out = torch.addmm(
            dconv1d_out, x_proj_weight.t(),
            dx_dbl[:, :delta_rank + d_state].t(),
            out=dconv1d_out
        )
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l",
                                b=x.shape[0], l=x.shape[-1])

        dqfc = torch.mm(q_proj_weight.t(), dx_dbl[:, -d_state:].t())
        dqfc = rearrange(dqfc, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])

        # Depthwise conv backward (signature: x, w, bias, dout, seq_idx, dx, has_bias).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_cuda.causal_conv1d_bwd(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, dx, ctx.has_conv1d_bias
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")

        return (dxz, dqfc,
                dconv1d_weight, dconv1d_bias,
                dx_proj_weight, dq_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, None, None, dD,                   # B, C are always None in input
                ddelta_bias if delta_bias is not None else None,
                dB_proj_bias, dC_proj_bias,
                None)                                  # delta_softplus (non-tensor)


def mamba_inner_fn(xz, qfc,
                   conv1d_weight, conv1d_bias,
                   x_proj_weight, q_proj_weight, delta_proj_weight,
                   out_proj_weight, out_proj_bias,
                   A, B=None, C=None, D=None,
                   delta_bias=None, B_proj_bias=None, C_proj_bias=None,
                   delta_softplus=True):
    """Cross-Mamba inner forward (fused CUDA kernel).

    Produces B from the context branch (xz) and C from the query branch (qfc),
    enabling a decoder-style interaction between the two sequences.
    """
    return MambaInnerFn.apply(
        xz, qfc,
        conv1d_weight, conv1d_bias,
        x_proj_weight, q_proj_weight, delta_proj_weight,
        out_proj_weight, out_proj_bias,
        A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus
    )