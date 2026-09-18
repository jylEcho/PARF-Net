# Adapted PARF-Net common modules for multi_phase framework.
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from swin_transformer_list import SwinTransformer  # type: ignore
    HAS_SWIN = True
except Exception:
    HAS_SWIN = False
    SwinTransformer = None


# -----------------------------
# Basic blocks
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, stride=stride),
            ConvBNReLU(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = DoubleConv(in_channels, out_channels, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.proj = ConvBNReLU(in_channels + skip_channels, out_channels, kernel_size=1)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.proj(x)
        return self.conv(x)


# -----------------------------
# LGAI
# -----------------------------
class LocalGlobalAttentionInteraction(nn.Module):
    """
    LGAI: Local-Global Attention Interaction Module
    local_feat: CNN局部特征
    global_feat: Swin全局特征
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.out_conv = DoubleConv(channels, channels)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        if local_feat.shape[-2:] != global_feat.shape[-2:]:
            global_feat = F.interpolate(global_feat, size=local_feat.shape[-2:], mode="bilinear", align_corners=False)

        spatial_mask = self.spatial_attn(local_feat)
        channel_weight = self.channel_attn(global_feat)
        guided_global = global_feat * spatial_mask * channel_weight
        fused = local_feat + guided_global
        return self.out_conv(fused)


# -----------------------------
# PARM
# -----------------------------
class PhaseAdaptiveRoutingModule(nn.Module):
    """
    PARM = phase routing + uncertainty estimation
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, use_uncertainty: bool = True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        if use_uncertainty:
            uh = max(hidden_dim // 2, 32)
            self.u_head = nn.Sequential(
                nn.Linear(in_dim, uh),
                nn.ReLU(inplace=True),
                nn.Linear(uh, 1),
                nn.Sigmoid(),
            )

    def _safe_vec(
        self,
        x: Optional[torch.Tensor],
        batch_size: int,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if x is None:
            return torch.zeros(batch_size, channels, device=device, dtype=dtype)
        return x

    def forward(
        self,
        g_a: Optional[torch.Tensor],
        g_p: Optional[torch.Tensor],
        g_d: Optional[torch.Tensor],
        present_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = present_mask.shape[0]
        device = present_mask.device
        dtype = present_mask.dtype

        channels = 0
        for item in [g_a, g_p, g_d]:
            if item is not None:
                channels = item.shape[1]
                dtype = item.dtype
                break
        if channels == 0:
            raise ValueError("At least one phase feature must be available.")

        g_a = self._safe_vec(g_a, batch_size, channels, device, dtype)
        g_p = self._safe_vec(g_p, batch_size, channels, device, dtype)
        g_d = self._safe_vec(g_d, batch_size, channels, device, dtype)

        logits = self.mlp(torch.cat([g_a, g_p, g_d], dim=1))
        neg_inf = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(present_mask == 0, neg_inf)
        alpha = torch.softmax(masked_logits, dim=1)

        if not self.use_uncertainty:
            return {"weights": alpha, "alpha": alpha, "uncertainty": torch.zeros_like(alpha)}

        u_a = self.u_head(g_a)
        u_p = self.u_head(g_p)
        u_d = self.u_head(g_d)
        uncertainty = torch.cat([u_a, u_p, u_d], dim=1)

        weights = alpha * (1.0 - uncertainty) * present_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return {"weights": weights, "alpha": alpha, "uncertainty": uncertainty}


# -----------------------------
# Backbone
# -----------------------------
class ConvPyramidFallback(nn.Module):
    """
    没有 swin_transformer_list.py 时
    """
    def __init__(self, in_channels: int, base_dim: int):
        super().__init__()
        self.l1 = DoubleConv(in_channels, base_dim)
        self.l2 = DownBlock(base_dim, base_dim * 2)
        self.l3 = DownBlock(base_dim * 2, base_dim * 4)
        self.l4 = DownBlock(base_dim * 4, base_dim * 8)

    def forward(self, x: torch.Tensor):
        f1 = self.l1(x)
        f2 = self.l2(f1)
        f3 = self.l3(f2)
        f4 = self.l4(f3)
        return [f1, f2, f3, f4]


class SwinBackboneAdapter(nn.Module):
    def __init__(self, in_channels: int, base_dim: int = 32, img_size: int = 224):
        super().__init__()
        self.use_real_swin = HAS_SWIN
        if self.use_real_swin:
            self.stem = ConvBNReLU(in_channels, base_dim, kernel_size=3)
            self.backbone = SwinTransformer(
                img_size=img_size,
                patch_size=4,
                in_chans=base_dim,
                num_classes=0,
                embed_dim=48,
                depths=[2, 2, 2, 1],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.1,
                patch_norm=True,
            )
            self.proj1 = nn.Sequential(nn.Conv2d(96, base_dim, 1), nn.BatchNorm2d(base_dim), nn.ReLU(inplace=True))
            self.proj2 = nn.Sequential(nn.Conv2d(192, base_dim * 2, 1), nn.BatchNorm2d(base_dim * 2), nn.ReLU(inplace=True))
            self.proj3 = nn.Sequential(nn.Conv2d(384, base_dim * 4, 1), nn.BatchNorm2d(base_dim * 4), nn.ReLU(inplace=True))
            self.proj4 = nn.Sequential(nn.Conv2d(384, base_dim * 8, 1), nn.BatchNorm2d(base_dim * 8), nn.ReLU(inplace=True))
        else:
            self.backbone = ConvPyramidFallback(in_channels, base_dim)

    @staticmethod
    def _reshape_tokens(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return x
        if x.ndim != 3:
            raise ValueError(f"Unexpected Swin output shape: {tuple(x.shape)}")
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = n // h
        return x.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x: torch.Tensor):
        if not self.use_real_swin:
            return self.backbone(x)

        x = self.stem(x)
        feats = self.backbone.forward_features(x)
        f1 = self.proj1(self._reshape_tokens(feats[0]))
        f2 = self.proj2(self._reshape_tokens(feats[1]))
        f3 = self.proj3(self._reshape_tokens(feats[2]))
        f4 = self.proj4(self._reshape_tokens(feats[3]))
        return [f1, f2, f3, f4]


# -----------------------------
# PARF-Net
# -----------------------------
class PARFNet(nn.Module):
    """
    PARF-Net with:
    - LGAI: Local-Global Attention Interaction Module
    - PARM: Phase Adaptive Routing Module
    - PCRM: Phase Consistency Regularization Module
    """
    def __init__(
        self,
        in_channels_per_phase: int = 1,
        num_classes: int = 1,
        base_dim: int = 32,
        img_size: int = 224,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.main_backbone = SwinBackboneAdapter(3 * in_channels_per_phase, base_dim, img_size)

        # three phase-specific CNN branches
        self.art_stem = DoubleConv(in_channels_per_phase, base_dim)
        self.pv_stem = DoubleConv(in_channels_per_phase, base_dim)
        self.dl_stem = DoubleConv(in_channels_per_phase, base_dim)

        self.art_down1 = DownBlock(base_dim, base_dim * 2)
        self.art_down2 = DownBlock(base_dim * 2, base_dim * 4)
        self.art_down3 = DownBlock(base_dim * 4, base_dim * 8)

        self.pv_down1 = DownBlock(base_dim, base_dim * 2)
        self.pv_down2 = DownBlock(base_dim * 2, base_dim * 4)
        self.pv_down3 = DownBlock(base_dim * 4, base_dim * 8)

        self.dl_down1 = DownBlock(base_dim, base_dim * 2)
        self.dl_down2 = DownBlock(base_dim * 2, base_dim * 4)
        self.dl_down3 = DownBlock(base_dim * 4, base_dim * 8)

        # LGAI at each scale
        self.lgai1 = LocalGlobalAttentionInteraction(base_dim)
        self.lgai2 = LocalGlobalAttentionInteraction(base_dim * 2)
        self.lgai3 = LocalGlobalAttentionInteraction(base_dim * 4)
        self.lgai4 = LocalGlobalAttentionInteraction(base_dim * 8)

        # PARM
        self.parm = PhaseAdaptiveRoutingModule(base_dim * 8, hidden_dim=256, use_uncertainty=use_uncertainty)

        # decoder
        self.up3 = UpBlock(base_dim * 8, base_dim * 4, base_dim * 4)
        self.up2 = UpBlock(base_dim * 4, base_dim * 2, base_dim * 2)
        self.up1 = UpBlock(base_dim * 2, base_dim, base_dim)
        self.seg_head = nn.Conv2d(base_dim, num_classes, kernel_size=1)

    @staticmethod
    def _infer_present_mask(
        art: torch.Tensor,
        pv: torch.Tensor,
        dl: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if present_mask is not None:
            return present_mask.float()

        def phase_flag(x: torch.Tensor) -> torch.Tensor:
            return (x.abs().flatten(1).sum(dim=1) > 0).float()

        mask = torch.stack([phase_flag(art), phase_flag(pv), phase_flag(dl)], dim=1).to(art.device)
        if mask.sum(dim=1).min().item() == 0:
            raise ValueError("At least one phase must be present for each sample.")
        return mask

    @staticmethod
    def _phase_encode(stem: nn.Module, d1: nn.Module, d2: nn.Module, d3: nn.Module, x: torch.Tensor):
        f1 = stem(x)
        f2 = d1(f1)
        f3 = d2(f2)
        f4 = d3(f3)
        return [f1, f2, f3, f4]

    @staticmethod
    def _weighted_phase_sum(features, weights: torch.Tensor, present_mask: torch.Tensor):
        out = 0.0
        for i, feat in enumerate(features):
            wi = (weights[:, i] * present_mask[:, i]).view(-1, 1, 1, 1)
            out = out + feat * wi
        return out

    def forward(
        self,
        art: torch.Tensor,
        pv: torch.Tensor,
        dl: torch.Tensor,
        present_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        present_mask = self._infer_present_mask(art, pv, dl, present_mask)

        # main branch
        main_input = torch.cat([art, pv, dl], dim=1)
        g1, g2, g3, g4 = self.main_backbone(main_input)

        # phase-specific branches
        art_feats = self._phase_encode(self.art_stem, self.art_down1, self.art_down2, self.art_down3, art)
        pv_feats = self._phase_encode(self.pv_stem, self.pv_down1, self.pv_down2, self.pv_down3, pv)
        dl_feats = self._phase_encode(self.dl_stem, self.dl_down1, self.dl_down2, self.dl_down3, dl)

        # PARM uses deepest pooled phase features
        def gap(x: torch.Tensor) -> torch.Tensor:
            return F.adaptive_avg_pool2d(x, 1).flatten(1)

        route_out = self.parm(
            gap(art_feats[-1]),
            gap(pv_feats[-1]),
            gap(dl_feats[-1]),
            present_mask,
        )
        weights = route_out["weights"]

        # mask-aware weighted auxiliary fusion
        aux1 = self._weighted_phase_sum([art_feats[0], pv_feats[0], dl_feats[0]], weights, present_mask)
        aux2 = self._weighted_phase_sum([art_feats[1], pv_feats[1], dl_feats[1]], weights, present_mask)
        aux3 = self._weighted_phase_sum([art_feats[2], pv_feats[2], dl_feats[2]], weights, present_mask)
        aux4 = self._weighted_phase_sum([art_feats[3], pv_feats[3], dl_feats[3]], weights, present_mask)

        # LGAI
        f1 = self.lgai1(aux1, g1)
        f2 = self.lgai2(aux2, g2)
        f3 = self.lgai3(aux3, g3)
        f4 = self.lgai4(aux4, g4)

        # decoder
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        logits = self.seg_head(d1)

        if return_aux:
            aux_dict = {
                "weights": route_out["weights"],
                "alpha": route_out["alpha"],
                "uncertainty": route_out["uncertainty"],
                "present_mask": present_mask,
                "fused_features": [f1, f2, f3, f4],
                "contrast_feature": F.adaptive_avg_pool2d(f4, 1).flatten(1),
                "prediction_logits": logits,
            }
            return logits, aux_dict
        return logits


# backward compatibility
HAformerSpatialFrequency = PARFNet


# -----------------------------
# Segmentation + PCRM losses
# -----------------------------
def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.3,
    bce_weight: float = 0.7,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = dice_loss_from_logits(logits, target)
    return bce_weight * bce + dice_weight * dice


def uncertainty_regularization(
    uncertainty: torch.Tensor,
    present_mask: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    reg = uncertainty * present_mask
    denom = present_mask.sum().clamp_min(1.0)
    if reduction == "sum":
        return reg.sum() / denom
    return reg.sum() / denom


def cosine_consistency_loss(
    full_logits: torch.Tensor,
    missing_logits: torch.Tensor,
    detach_teacher: bool = True,
) -> torch.Tensor:
    full_prob = torch.sigmoid(full_logits)
    miss_prob = torch.sigmoid(missing_logits)

    if detach_teacher:
        full_prob = full_prob.detach()

    full_vec = full_prob.flatten(1)
    miss_vec = miss_prob.flatten(1)
    return 1.0 - F.cosine_similarity(miss_vec, full_vec, dim=1).mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard supervised contrastive loss.
    features: [N, C]
    labels:   [N]
    """
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature

    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = mask.sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / pos_count
    return -mean_log_prob_pos.mean()


def batchwise_full_missing_supcon_loss(
    full_features: torch.Tensor,
    missing_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    正样本：同一病例的 full / missing 深层表示
    负样本：同一 mini-batch 内其他病例的 full / missing 表示
    """
    if full_features.shape != missing_features.shape:
        raise ValueError("full_features and missing_features must have the same shape.")

    batch_size = full_features.shape[0]
    all_features = torch.cat([full_features, missing_features], dim=0)
    labels = torch.arange(batch_size, device=full_features.device)
    labels = torch.cat([labels, labels], dim=0)
    return supervised_contrastive_loss(all_features, labels, temperature=temperature)


def compute_total_loss(
    full_logits: torch.Tensor,
    missing_logits: torch.Tensor,
    target: torch.Tensor,
    aux_full: Dict[str, torch.Tensor],
    aux_missing: Dict[str, torch.Tensor],
    lambda_uncertainty: float = 0.05,
    lambda_supcon: float = 0.1,
    lambda_cosine: float = 0.2,
    temperature: float = 0.07,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    PCRM = SupCon + Cosine consistency
    训练方式：
    1) 同一个 batch 跑完整三期 forward -> full_logits, aux_full
    2) 再构造缺一期输入 forward -> missing_logits, aux_missing
    3) 调用本函数组装总 loss
    """
    seg = segmentation_loss(missing_logits, target, pos_weight=pos_weight)
    unc = uncertainty_regularization(aux_missing["uncertainty"], aux_missing["present_mask"])
    supcon = batchwise_full_missing_supcon_loss(
        aux_full["contrast_feature"],
        aux_missing["contrast_feature"],
        temperature=temperature,
    )
    cos = cosine_consistency_loss(full_logits, missing_logits, detach_teacher=True)

    total = seg + lambda_uncertainty * unc + lambda_supcon * supcon + lambda_cosine * cos
    return {
        "loss_total": total,
        "loss_seg": seg,
        "loss_uncertainty": unc,
        "loss_supcon": supcon,
        "loss_cosine": cos,
    }


if __name__ == "__main__":
    model = PARFNet(in_channels_per_phase=1, num_classes=1, base_dim=32, img_size=224)

    art = torch.randn(2, 1, 224, 224)
    pv = torch.randn(2, 1, 224, 224)
    dl = torch.randn(2, 1, 224, 224)

    # full phase
    full_logits, full_aux = model(art, pv, dl, return_aux=True)
    print("full logits:", full_logits.shape)

    # missing PV example
    missing_mask = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.float32)
    miss_logits, miss_aux = model(art, torch.zeros_like(pv), dl, present_mask=missing_mask, return_aux=True)
    print("missing logits:", miss_logits.shape)

    target = torch.randint(0, 2, (2, 1, 224, 224)).float()
    losses = compute_total_loss(full_logits, miss_logits, target, full_aux, miss_aux)
    print({k: float(v.detach().cpu()) for k, v in losses.items()})