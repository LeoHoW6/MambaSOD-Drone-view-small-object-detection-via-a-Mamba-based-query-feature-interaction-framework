import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import List, Optional, Dict

from mamba_ssm.modules.mamba_simple import Mamba
from cross_mamba import Mamba as Cross_Mamba
from mamba_block import CrossBlock, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None

from encoder import MambaSODEncoder
from ablation_modules import StandardCA, StandardSA


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


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


# ==================== MQSI ====================

class MQSI(nn.Module):
    def __init__(self, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba_fw = Block(
            d_model,
            mixer_cls=partial(Mamba, d_state=d_state, d_conv=d_conv, expand=expand),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )
        self.mamba_bw = Block(
            d_model,
            mixer_cls=partial(Mamba, d_state=d_state, d_conv=d_conv, expand=expand),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )
        self.fuse = nn.Linear(d_model * 2, d_model)

    def forward(self, queries: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
        residual = queries
        B, total_Nq, D = queries.shape
        assert total_Nq % num_groups == 0
        Nq = total_Nq // num_groups

        if num_groups > 1:
            q_grouped = queries.reshape(B, num_groups, Nq, D)
            q_grouped = q_grouped.reshape(B * num_groups, Nq, D)
        else:
            q_grouped = queries

        fw_out, _ = self.mamba_fw(q_grouped, residual=None)
        bw_out, _ = self.mamba_bw(q_grouped.flip([1]), residual=None)
        bw_out = bw_out.flip([1])

        fused = self.fuse(torch.cat([fw_out, bw_out], dim=-1))

        if num_groups > 1:
            fused = fused.reshape(B, num_groups, Nq, D)
            fused = fused.reshape(B, num_groups * Nq, D)

        return residual + fused


# ==================== CrossMamba ====================

class CrossMambaModule(nn.Module):
    def __init__(self, d_model: int = 256, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.cross_fw = CrossBlock(
            d_model,
            mixer_cls=partial(Cross_Mamba, d_state=d_state,
                              d_conv=d_conv, expand=expand),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )
        self.cross_bw = CrossBlock(
            d_model,
            mixer_cls=partial(Cross_Mamba, d_state=d_state,
                              d_conv=d_conv, expand=expand),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )

    def forward(self, queries: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        fw_out, fw_res = self.cross_fw(queries, features, residual=None)
        forward_f = fw_out + fw_res

        bw_out, bw_res = self.cross_bw(
            queries.flip([1]), features.flip([1]), residual=None
        )
        backward_f = (bw_out + bw_res).flip([1])

        return forward_f + backward_f


# ==================== MQI Scale Branch ====================

class MQIScaleBranch(nn.Module):
    def __init__(self, d_model: int, num_queries: int = 400,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_queries_per_group = num_queries

        side = int(round(math.sqrt(num_queries)))
        assert side * side == num_queries
        self.target_side = side

        self.feat_refine = Block(
            d_model,
            mixer_cls=partial(Mamba, d_state=d_state, d_conv=d_conv, expand=expand),
            norm_cls=partial(RMSNorm, eps=1e-5),
            fused_add_norm=False,
        )

        self.cross_q_reads_f = CrossMambaModule(
            d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def align_to_grid_2d(self, feat: torch.Tensor, in_hw: tuple) -> torch.Tensor:
        B, L, D = feat.shape
        in_H, in_W = in_hw
        assert L == in_H * in_W

        target_H = target_W = self.target_side

        if (in_H, in_W) == (target_H, target_W):
            return feat

        feat_2d = feat.transpose(1, 2).reshape(B, D, in_H, in_W).contiguous()

        if in_H >= target_H and in_W >= target_W:
            if in_H % target_H == 0 and in_W % target_W == 0:
                kh = in_H // target_H
                kw = in_W // target_W
                feat_2d = F.avg_pool2d(feat_2d, kernel_size=(kh, kw),
                                        stride=(kh, kw))
            else:
                feat_2d = F.adaptive_avg_pool2d(feat_2d, (target_H, target_W))
        else:
            feat_2d = F.interpolate(feat_2d, size=(target_H, target_W),
                                     mode='bilinear', align_corners=False)

        return feat_2d.flatten(2).transpose(1, 2)

    def forward(self, queries: torch.Tensor, feat: torch.Tensor,
                feat_hw: tuple, num_groups: int = 1) -> torch.Tensor:
        B, total_Nq, D = queries.shape
        Nq = self.num_queries_per_group

        feat_refined, feat_res = self.feat_refine(feat, residual=None)
        feat_refined = feat_refined + feat_res

        feat_aligned_per_group = self.align_to_grid_2d(feat_refined, feat_hw)

        if num_groups > 1:
            feat_aligned = feat_aligned_per_group.unsqueeze(1).expand(
                B, num_groups, Nq, D
            ).reshape(B * num_groups, Nq, D).contiguous()
            q_grouped = queries.reshape(B, num_groups, Nq, D).reshape(
                B * num_groups, Nq, D
            ).contiguous()
        else:
            feat_aligned = feat_aligned_per_group
            q_grouped = queries

        out = self.cross_q_reads_f(q_grouped, feat_aligned)
        out = self.out_proj(out)

        if num_groups > 1:
            out = out.reshape(B, num_groups, Nq, D).reshape(B, num_groups * Nq, D)

        return out


class MQI(nn.Module):
    def __init__(self, d_model: int, num_queries: int = 400,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 feature_shapes: Optional[List[tuple]] = None,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_scales = len(feature_shapes) if feature_shapes else 4

        if feature_shapes is None:
            feature_shapes = [(80, 80), (40, 40), (20, 20), (10, 10)]
        self.feature_shapes = feature_shapes

        self.scale_branches = nn.ModuleList([
            MQIScaleBranch(
                d_model=d_model, num_queries=num_queries,
                d_state=d_state, d_conv=d_conv, expand=expand,
            )
            for _ in feature_shapes
        ])

        self.scale_gate = nn.Linear(d_model * self.num_scales, self.num_scales)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor,
                memories: List[torch.Tensor],
                pos_embeds: List[torch.Tensor],
                num_groups: int = 1) -> torch.Tensor:
        residual = queries
        queries_normed = self.norm(queries)

        scale_outputs = []
        for i in range(self.num_scales):
            feat = memories[i] + pos_embeds[i]
            out = self.scale_branches[i](
                queries_normed, feat,
                feat_hw=self.feature_shapes[i],
                num_groups=num_groups,
            )
            scale_outputs.append(out)

        fused = torch.stack(scale_outputs, dim=0).mean(dim=0)

        return residual + self.dropout(fused)


# ==================== FFN ====================

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear2(self.dropout1(self.act(self.linear1(x))))
        return residual + self.dropout2(x)


# ==================== Decoder Layer ====================

class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model=256, num_queries=300,
                d_state=16, d_conv=4, expand=2,
                d_ffn=1024, dropout=0.0,
                feature_shapes=None,
                use_ca=False, use_sa=False):
        super().__init__()
        if use_sa:
            self.mqsi = StandardSA(d_model=d_model, dropout=dropout)
        else:
            self.mqsi = MQSI(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        if use_ca:
            self.mqi = StandardCA(d_model=d_model, num_queries=num_queries,
                                feature_shapes=feature_shapes, dropout=dropout)
        else:
            self.mqi = MQI(d_model, num_queries=num_queries,
                            d_state=d_state, d_conv=d_conv, expand=expand,
                            feature_shapes=feature_shapes, dropout=dropout)
        self.ffn = FFN(d_model, d_ffn=d_ffn, dropout=dropout)

    def forward(self, tgt, query_pos, memories, pos_embeds, num_groups: int = 1):
        q = tgt + query_pos
        tgt = self.mqsi(q, num_groups=num_groups)

        q = tgt + query_pos
        tgt = self.mqi(q, memories, pos_embeds, num_groups=num_groups)

        tgt = self.ffn(tgt)
        return tgt


# ==================== MLP ====================

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.act = nn.GELU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ==================== Decoder ====================

class MambaSODDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 400,
        num_decoder_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_ffn: int = 1024,
        num_classes: int = 10,
        dropout: float = 0.0,
        feature_shapes: Optional[List[tuple]] = None,
        num_groups: int = 4,
        use_ca=False,
        use_sa=False,
    ):
        super().__init__()

        side = int(round(math.sqrt(num_queries)))
        assert side * side == num_queries

        self.d_model = d_model
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.num_decoder_layers = num_decoder_layers
        self.num_groups_train = num_groups

        if feature_shapes is None:
            feature_shapes = [(80, 80), (40, 40), (20, 20), (10, 10)]

        self.tgt_embed = nn.Embedding(num_groups * num_queries, d_model)
        self.query_pos = nn.Embedding(num_groups * num_queries, d_model)
        self.reference_point = nn.Embedding(num_queries, 4)

        self.layers = nn.ModuleList([
            MambaDecoderLayer(
                d_model=d_model, num_queries=num_queries,
                d_state=d_state, d_conv=d_conv, expand=expand,
                d_ffn=d_ffn, dropout=dropout,
                feature_shapes=feature_shapes,
                use_ca=use_ca, use_sa=use_sa,
            )
            for _ in range(num_decoder_layers)
        ])

        self.norm = RMSNorm(d_model)

        self.class_heads = nn.ModuleList([
            nn.Linear(d_model, num_classes) for _ in range(num_decoder_layers)
        ])
        self.bbox_heads = nn.ModuleList([
            MLP(d_model, d_model, 4, num_layers=3) for _ in range(num_decoder_layers)
        ])

        self._init_weights()

    def _init_weights(self):
        side = int(round(math.sqrt(self.num_queries)))
        with torch.no_grad():
            grid_coord = torch.linspace(0.05, 0.95, side)
            cy, cx = torch.meshgrid(grid_coord, grid_coord, indexing='ij')
            init_cxcy = torch.stack([cx.flatten(), cy.flatten()], dim=-1)

            wh_log = torch.randn(self.num_queries, 2) * 0.7 + math.log(0.04)
            init_wh = wh_log.exp().clamp(0.008, 0.5)

            init_boxes = torch.cat([init_cxcy, init_wh], dim=-1)
            init_boxes = init_boxes.clamp(0.005, 0.995)
            self.reference_point.weight.data = inverse_sigmoid(init_boxes)

        nn.init.normal_(self.tgt_embed.weight, std=0.1)
        nn.init.normal_(self.query_pos.weight, std=0.1)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_heads:
            nn.init.constant_(class_head.bias, bias_value)

        for bbox_head in self.bbox_heads:
            nn.init.constant_(bbox_head.layers[-1].weight, 0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0)

    def forward(self, encoder_output: Dict) -> Dict:
        memories = encoder_output['memories']
        pos_embeds = encoder_output['pos_embeds']
        B = memories[0].shape[0]

        if self.training:
            num_groups = self.num_groups_train
            tgt = self.tgt_embed.weight.unsqueeze(0).expand(B, -1, -1)
            query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        else:
            num_groups = 1
            tgt = self.tgt_embed.weight[:self.num_queries].unsqueeze(0).expand(B, -1, -1)
            query_pos = self.query_pos.weight[:self.num_queries].unsqueeze(0).expand(B, -1, -1)

        ref_points_logit = self.reference_point.weight
        if num_groups > 1:
            ref_points_logit = ref_points_logit.unsqueeze(0).repeat(num_groups, 1, 1).reshape(
                num_groups * self.num_queries, 4
            )
        ref_points_logit = ref_points_logit.unsqueeze(0).expand(B, -1, -1)

        aux_outputs = []
        for layer_idx, layer in enumerate(self.layers):
            tgt = layer(tgt, query_pos, memories, pos_embeds, num_groups=num_groups)
            normed = self.norm(tgt)

            pred_logits = self.class_heads[layer_idx](normed)
            delta = self.bbox_heads[layer_idx](normed + query_pos)
            pred_boxes = (ref_points_logit + delta).sigmoid()

            aux_outputs.append({
                'pred_logits': pred_logits,
                'pred_boxes': pred_boxes,
            })

        final = aux_outputs[-1]
        return {
            'pred_logits': final['pred_logits'],
            'pred_boxes': final['pred_boxes'],
            'aux_outputs': aux_outputs[:-1],
            'num_groups': num_groups,
        }


# ==================== MambaSOD ====================

class MambaSOD(nn.Module):
    def __init__(
        self,
        img_size=640, patch_size=16, vim_depth=12, vim_embed_dim=192,
        d_state=16, out_channels=256, bifpn_repeats=3,
        num_queries=400, num_decoder_layers=6, d_ffn=1024,
        num_classes=10, dropout=0.1, drop_path_rate=0.1,
        num_groups=4,
        use_ca=False, use_sa=False, use_fpn=False,
    ):
        super().__init__()
        self.encoder = MambaSODEncoder(
            img_size=img_size, patch_size=patch_size,
            vim_depth=vim_depth, vim_embed_dim=vim_embed_dim,
            d_state=d_state, out_channels=out_channels,
            bifpn_repeats=bifpn_repeats, drop_path_rate=drop_path_rate,
            use_fpn=use_fpn,
        )
        grid = img_size // patch_size
        feature_shapes = [
            (grid * 4, grid * 4),
            (grid * 2, grid * 2),
            (grid, grid),
            (grid // 2, grid // 2),
            (grid // 4, grid // 4),
        ]
        self.decoder = MambaSODDecoder(
            d_model=out_channels, num_queries=num_queries,
            num_decoder_layers=num_decoder_layers, d_state=d_state,
            d_ffn=d_ffn, num_classes=num_classes, dropout=dropout,
            feature_shapes=feature_shapes,
            num_groups=num_groups,
            use_ca=use_ca,
            use_sa=use_sa,
        )

    def forward(self, images):
        return self.decoder(self.encoder(images))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MambaSOD(
        img_size=640, patch_size=16, vim_depth=24, vim_embed_dim=192,
        d_state=16, out_channels=256, bifpn_repeats=3,
        num_queries=400, num_decoder_layers=6, d_ffn=1024,
        num_classes=10, dropout=0.1, num_groups=4,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    model.eval()
    images = torch.randn(2, 3, 640, 640, device=device)
    with torch.no_grad():
        out = model(images)
    print(f"logits: {tuple(out['pred_logits'].shape)}")
    print(f"boxes:  {tuple(out['pred_boxes'].shape)}")