import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Optional

from timm.models.layers import trunc_normal_, DropPath
from vim.models_mamba import (
    PatchEmbed,
    create_block,
    _init_weights,
    segm_init_weights,
)
from ablation_modules import StandardFPN

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


if RMSNorm is None:
    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5, **kwargs):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(hidden_size))

        def forward(self, x, **kwargs):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return (x * self.weight).to(dtype=x.dtype)

    layer_norm_fn = None
    rms_norm_fn = None


# ==================== Multi-Scale Projection ====================

class MultiScaleProjection(nn.Module):
    def __init__(self, embed_dim: int, out_channels: int, scale_type: str):
        super().__init__()

        if scale_type == 'up2x':
            self.proj = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, out_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        elif scale_type == 'keep':
            self.proj = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        elif scale_type == 'down2x':
            self.proj = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        elif scale_type == 'down4x':
            self.proj = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown scale_type: {scale_type}")

        self.apply(segm_init_weights)

    def forward(self, x):
        return self.proj(x)


# ==================== VisionMamba Backbone ====================

class VisionMambaBackbone(nn.Module):
    def __init__(
        self,
        img_size: int = 640,
        patch_size: int = 16,
        depth: int = 12,
        embed_dim: int = 192,
        d_state: int = 16,
        out_channels: int = 256,
        channels: int = 3,
        ssm_cfg: Optional[dict] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = True,
        initializer_cfg: Optional[dict] = None,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        if_bidirectional: bool = False,
        if_bimamba: bool = False,
        bimamba_type: str = "v2",
        if_cls_token: bool = True,
        if_divide_out: bool = True,
        use_middle_cls_token: bool = True,
        init_layer_scale: Optional[float] = None,
        if_abs_pos_embed: bool = True,
        if_rope: bool = False,
        if_rope_residual: bool = False,
        flip_img_sequences_ratio: float = -1.,
        pt_hw_seq_len: int = 14,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.depth = depth
        self.residual_in_fp32 = residual_in_fp32
        self.if_bidirectional = if_bidirectional
        self.if_cls_token = if_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.num_tokens = 1 if if_cls_token else 0

        if fused_add_norm and rms_norm_fn is None:
            fused_add_norm = False
        self.fused_add_norm = fused_add_norm

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=patch_size,
            in_chans=channels, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.grid_size = self.patch_embed.grid_size

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)
            trunc_normal_(self.pos_embed, std=.02)

        if if_rope:
            from rope import VisionRotaryEmbeddingFast
            half_head_dim = embed_dim // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len,
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                d_state=d_state,
                ssm_cfg=ssm_cfg if ssm_cfg is not None else {},
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_divide_out=if_divide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs,
            )
            for i in range(depth)
        ])

        if depth >= 24:
            self.extract_layers = [2, 8, 15, 23]
        else:
            interval = depth // 4
            self.extract_layers = [
                interval - 1, interval * 2 - 1,
                interval * 3 - 1, depth - 1,
            ]

        NormClass = RMSNorm if rms_norm else partial(nn.LayerNorm, **factory_kwargs)
        self.intermediate_norms = nn.ModuleList([
            NormClass(embed_dim, eps=norm_epsilon)
            for _ in range(4)
        ])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.multi_scale_projs = nn.ModuleList([
            MultiScaleProjection(embed_dim, out_channels, 'up2x'),
            MultiScaleProjection(embed_dim, out_channels, 'keep'),
            MultiScaleProjection(embed_dim, out_channels, 'down2x'),
            MultiScaleProjection(embed_dim, out_channels, 'down4x'),
        ])

        self.patch_embed.apply(segm_init_weights)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _extract_feature(self, hidden_states, residual, norm, token_position):
        B = hidden_states.shape[0]

        if residual is not None:
            combined = residual + hidden_states
        else:
            combined = hidden_states

        feat = norm(combined.to(dtype=norm.weight.dtype))

        if self.if_cls_token:
            if self.use_middle_cls_token:
                feat_no_cls = torch.cat([
                    feat[:, :token_position, :],
                    feat[:, token_position + 1:, :]
                ], dim=1)
            else:
                feat_no_cls = feat[:, 1:, :]
        else:
            feat_no_cls = feat

        H, W = self.grid_size
        feat_2d = feat_no_cls.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        return feat_2d

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        import random

        B = x.shape[0]

        x = self.patch_embed(x)
        M = x.shape[1]

        token_position = -1
        if self.if_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.use_middle_cls_token:
                token_position = M // 2
                x = torch.cat([
                    x[:, :token_position, :],
                    cls_token,
                    x[:, token_position:, :]
                ], dim=1)
            else:
                token_position = 0
                x = torch.cat([cls_token, x], dim=1)
            M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if_flip_img_sequences = False
        if self.training and self.flip_img_sequences_ratio > 0:
            if (self.flip_img_sequences_ratio - random.random()) > 1e-5:
                x = x.flip([1])
                if_flip_img_sequences = True

        residual = None
        hidden_states = x
        intermediate_features = []
        extract_idx = 0

        if not self.if_bidirectional:
            for i, layer in enumerate(self.layers):

                if self.if_rope:
                    if if_flip_img_sequences:
                        hidden_states = hidden_states.flip([1])
                        if residual is not None:
                            residual = residual.flip([1])

                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                    if if_flip_img_sequences:
                        hidden_states = hidden_states.flip([1])
                        if residual is not None:
                            residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None
                )

                if extract_idx < 4 and i == self.extract_layers[extract_idx]:
                    feat_2d = self._extract_feature(
                        hidden_states, residual,
                        self.intermediate_norms[extract_idx],
                        token_position,
                    )
                    intermediate_features.append(feat_2d)
                    extract_idx += 1
        else:
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=None
                )
                hidden_states_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                    inference_params=None
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])

                layer_idx = i * 2 + 1
                if extract_idx < 4 and layer_idx == self.extract_layers[extract_idx]:
                    feat_2d = self._extract_feature(
                        hidden_states, residual,
                        self.intermediate_norms[extract_idx],
                        token_position,
                    )
                    intermediate_features.append(feat_2d)
                    extract_idx += 1

        assert len(intermediate_features) == 4

        multi_scale_features = [
            proj(feat) for proj, feat
            in zip(self.multi_scale_projs, intermediate_features)
        ]

        return multi_scale_features


# ==================== BiFPN ====================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, num_channels, num_levels=4, eps=1e-4):
        super().__init__()
        self.num_levels = num_levels
        self.eps = eps

        self.w_td = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(num_levels - 1)
        ])
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels)
            for _ in range(num_levels - 1)
        ])

        self.w_bu = nn.ParameterList()
        for i in range(num_levels - 1):
            n_inputs = 2 if i == num_levels - 2 else 3
            self.w_bu.append(
                nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
            )
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels)
            for _ in range(num_levels - 1)
        ])

    def _fuse(self, weights, features):
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)
        th, tw = features[0].shape[-2:]
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            fh, fw = feat.shape[-2:]
            if (fh, fw) != (th, tw):
                if fh > th or fw > tw:
                    kh = max(1, fh // th)
                    kw = max(1, fw // tw)
                    feat = F.max_pool2d(feat, kernel_size=(kh, kw), stride=(kh, kw))
                    if feat.shape[-2:] != (th, tw):
                        feat = F.adaptive_max_pool2d(feat, (th, tw))
                else:
                    feat = F.interpolate(feat, size=(th, tw),
                                         mode='bilinear', align_corners=False)
            fused = fused + w[i] * feat
        return fused

    def forward(self, features):
        N = self.num_levels

        td = [None] * N
        td[-1] = features[-1]
        for i in range(N - 2, -1, -1):
            td[i] = self.td_convs[i](
                self._fuse(self.w_td[i], [features[i], td[i + 1]])
            )

        bu = [None] * N
        bu[0] = td[0]
        for i in range(1, N):
            if i == N - 1:
                inputs = [td[i], bu[i - 1]]
            else:
                inputs = [features[i], td[i], bu[i - 1]]
            bu[i] = self.bu_convs[i - 1](
                self._fuse(self.w_bu[i - 1], inputs)
            )

        return bu


class BiFPN(nn.Module):
    def __init__(self, in_channels: int = 256,
                 num_levels: int = 4, num_repeats: int = 3):
        super().__init__()
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(in_channels, num_levels)
            for _ in range(num_repeats)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        for layer in self.bifpn_layers:
            features = layer(features)
        return features


# ==================== 2D Positional Encoding ====================

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_shape=(100, 100)):
        super().__init__()
        self.d_model = d_model
        self._build_pe(*max_shape)

    def _build_pe(self, max_h, max_w):
        pe = torch.zeros(max_h, max_w, self.d_model)
        d_half = self.d_model // 2

        y = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        x = torch.arange(0, max_w, dtype=torch.float).unsqueeze(0)
        div = torch.exp(
            torch.arange(0, d_half, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_half)
        )

        pe[:, :, 0:d_half:2] = torch.sin(
            y.unsqueeze(2) * div
        ).expand(-1, max_w, -1)
        pe[:, :, 1:d_half:2] = torch.cos(
            y.unsqueeze(2) * div
        ).expand(-1, max_w, -1)
        pe[:, :, d_half::2] = torch.sin(
            x.unsqueeze(2) * div
        ).expand(max_h, -1, -1)
        pe[:, :, d_half + 1::2] = torch.cos(
            x.unsqueeze(2) * div
        ).expand(max_h, -1, -1)

        self.register_buffer('pe', pe)
        self.max_h = max_h
        self.max_w = max_w

    def forward(self, B, H, W, device):
        if H > self.max_h or W > self.max_w:
            self._build_pe(max(H, self.max_h), max(W, self.max_w))
            self.pe = self.pe.to(device)
        pe = self.pe[:H, :W, :].reshape(H * W, self.d_model)
        return pe.unsqueeze(0).expand(B, -1, -1).to(device)


# ==================== MambaSOD Encoder ====================

class MambaSODEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 640,
        patch_size: int = 16,
        vim_depth: int = 12,
        vim_embed_dim: int = 192,
        d_state: int = 16,
        out_channels: int = 256,
        bifpn_repeats: int = 3,
        drop_path_rate: float = 0.1,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        residual_in_fp32: bool = True,
        if_bidirectional: bool = False,
        if_bimamba: bool = False,
        bimamba_type: str = "v2",
        if_cls_token: bool = True,
        if_divide_out: bool = True,
        use_middle_cls_token: bool = True,
        if_abs_pos_embed: bool = True,
        if_rope: bool = False,
        if_rope_residual: bool = False,
        flip_img_sequences_ratio: float = -1.,
        use_fpn=False,
        init_layer_scale: Optional[float] = None,
    ):
        super().__init__()
        self.out_channels = out_channels

        self.backbone = VisionMambaBackbone(
            img_size=img_size,
            patch_size=patch_size,
            depth=vim_depth,
            embed_dim=vim_embed_dim,
            d_state=d_state,
            out_channels=out_channels,
            drop_path_rate=drop_path_rate,
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            if_bidirectional=if_bidirectional,
            if_bimamba=if_bimamba,
            bimamba_type=bimamba_type,
            if_cls_token=if_cls_token,
            if_divide_out=if_divide_out,
            use_middle_cls_token=use_middle_cls_token,
            if_abs_pos_embed=if_abs_pos_embed,
            if_rope=if_rope,
            if_rope_residual=if_rope_residual,
            flip_img_sequences_ratio=flip_img_sequences_ratio,
            init_layer_scale=init_layer_scale,
        )

        # -------- Efficient P-1 branch (stride=4) --------
        self.p_minus1_stem = nn.Sequential(
            nn.Conv2d(3, out_channels // 4, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU(),
            nn.Conv2d(out_channels // 4, out_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        if use_fpn:
            self.neck = StandardFPN(
                in_channels=out_channels,
                num_levels=5,
            )
        else:
            self.neck = BiFPN(
                in_channels=out_channels,
                num_levels=5,
                num_repeats=bifpn_repeats,
            )

        self.pos_encoding = PositionalEncoding2D(out_channels)

    def forward(self, x: torch.Tensor) -> dict:
        B = x.shape[0]

        # P-1 directly from image (stride=4, 160x160)
        p_minus1 = self.p_minus1_stem(x)

        # ViM -> 4 scales: P0(80), P1(40), P2(20), P3(10)
        multi_scale = self.backbone(x)

        # Assemble 5-level pyramid
        multi_scale = [p_minus1] + multi_scale
        enhanced = self.neck(multi_scale)

        memories, pos_embeds = [], []
        for feat in enhanced:
            _, C, H, W = feat.shape
            print(f"  Feature {tuple(feat.shape)}")
            memories.append(feat.flatten(2).transpose(1, 2))
            pos_embeds.append(self.pos_encoding(B, H, W, feat.device))

        return {'features': enhanced, 'memories': memories, 'pos_embeds': pos_embeds}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = MambaSODEncoder(
        img_size=640, patch_size=16, vim_depth=12, vim_embed_dim=192,
        d_state=16, out_channels=256, bifpn_repeats=3,
    ).to(device)

    total = sum(p.numel() for p in encoder.parameters())
    print(f"Total params: {total:,}")

    x = torch.randn(2, 3, 640, 640).to(device)
    encoder.eval()
    with torch.no_grad():
        out = encoder(x)