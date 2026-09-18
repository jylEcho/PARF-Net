import torch
import torch.nn as nn

try:
    from .parfnet_common import PARFNet, HAS_SWIN
except ImportError:
    from parfnet_common import PARFNet, HAS_SWIN


class Network(nn.Module):
    """
    PARF-Net 3P adapter for the existing multi_phase framework.

    External interface used by your trainers/test_missing_3p.py:
        forward(image_v, image_a, image_d)

    Internal PARFNet order:
        forward(art, pv, dl)
    """
    def __init__(
        self,
        num_classes=3,
        base_dim=32,
        img_size=224,
        in_channels_per_phase=1,
        use_uncertainty=True,
    ):
        super().__init__()
        self.backbone = PARFNet(
            in_channels_per_phase=in_channels_per_phase,
            num_classes=num_classes,
            base_dim=base_dim,
            img_size=img_size,
            use_uncertainty=use_uncertainty,
        )

    def forward(self, image_v, image_a, image_d):
        # External order: V, A, D. Internal order: A, V, D.
        return self.backbone(image_a, image_v, image_d, present_mask=None, return_aux=False)


if __name__ == "__main__":
    model = Network(num_classes=3, base_dim=4, img_size=64)
    v = torch.randn(1, 1, 64, 64)
    a = torch.randn(1, 1, 64, 64)
    d = torch.randn(1, 1, 64, 64)
    y = model(v, a, d)
    print("HAS_SWIN =", HAS_SWIN)
    print("output:", y.shape)
