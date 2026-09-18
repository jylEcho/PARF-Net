import torch
import torch.nn as nn

try:
    from .parfnet_common import PARFNet, HAS_SWIN
except ImportError:
    from parfnet_common import PARFNet, HAS_SWIN


class Network(nn.Module):
    """
    PARF-Net 2P adapter for the existing multi_phase framework.

    External training interface:
        forward(x1, x2)

    The phase identities are controlled by phase_combo. Default is ("D", "A")
    to match your PA-Net/2P ART+DL experiment command: --phase_combo D A.

    Internally PARFNet expects:
        forward(art, pv, dl, present_mask)
    where present_mask order is [A, V, D].
    """
    def __init__(
        self,
        num_classes=3,
        base_dim=32,
        img_size=224,
        in_channels_per_phase=1,
        use_uncertainty=True,
        phase_combo=("D", "A"),
    ):
        super().__init__()
        self.backbone = PARFNet(
            in_channels_per_phase=in_channels_per_phase,
            num_classes=num_classes,
            base_dim=base_dim,
            img_size=img_size,
            use_uncertainty=use_uncertainty,
        )
        self.phase_combo = [self._norm_phase(p) for p in phase_combo]

    @staticmethod
    def _norm_phase(p):
        p = str(p).strip().upper()
        if p not in ["A", "V", "D"]:
            raise ValueError(f"Unsupported phase: {p}. Expected one of A/V/D.")
        return p

    def set_phase_combo(self, phase_combo):
        if len(phase_combo) != 2:
            raise ValueError(f"PARFNet 2P expects exactly two phases, got {phase_combo}")
        self.phase_combo = [self._norm_phase(p) for p in phase_combo]

    def _build_inputs(self, x1, x2):
        zero = torch.zeros_like(x1)
        phase_map = {"A": zero.clone(), "V": zero.clone(), "D": zero.clone()}
        phase_map[self.phase_combo[0]] = x1
        phase_map[self.phase_combo[1]] = x2

        b = x1.shape[0]
        present_mask = torch.zeros(b, 3, device=x1.device, dtype=x1.dtype)
        idx = {"A": 0, "V": 1, "D": 2}
        present_mask[:, idx[self.phase_combo[0]]] = 1.0
        present_mask[:, idx[self.phase_combo[1]]] = 1.0
        return phase_map["A"], phase_map["V"], phase_map["D"], present_mask

    def forward(self, x1, x2):
        art, pv, dl, present_mask = self._build_inputs(x1, x2)
        return self.backbone(art, pv, dl, present_mask=present_mask, return_aux=False)


if __name__ == "__main__":
    model = Network(num_classes=3, base_dim=4, img_size=64, phase_combo=("D", "A"))
    x1 = torch.randn(1, 1, 64, 64)
    x2 = torch.randn(1, 1, 64, 64)
    y = model(x1, x2)
    print("HAS_SWIN =", HAS_SWIN)
    print("output:", y.shape)
