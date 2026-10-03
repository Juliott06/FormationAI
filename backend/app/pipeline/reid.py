"""Person re-identification embeddings (OSNet x1.0, MSMT17-trained).

Colour histograms can't separate dancers who all wear black, and generic
vision features (DINOv2) mostly encode pose. OSNet is trained specifically to
recognise the same person across cameras, poses and scales, from clothing
details — what identity matching needs.

The architecture below follows OSNet (Zhou et al., "Omni-Scale Feature
Learning for Person Re-Identification", ICCV 2019) with parameter names
matching the torchreid checkpoints, so the published MSMT17 weights load
directly. Weights are downloaded once to the torch hub cache.
"""
from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger("uvicorn.error")

# Published torchreid MSMT17 checkpoints (Google Drive), by width multiplier.
VARIANTS = {
    "x1_0": ((64, 256, 384, 512), "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"),
    "x0_75": ((48, 192, 288, 384), "https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc"),
    "x0_5": ((32, 128, 192, 256), "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv"),
    "x0_25": ((16, 64, 96, 128), "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"),
}
VARIANT = "x0_25"  # best on the real 7-dancer clip AND ~7x faster than x1_0
_INPUT_HW = (256, 128)

_model = None


def _build(channels):
    import torch
    from torch import nn

    class ConvLayer(nn.Module):
        def __init__(self, cin, cout, k, stride=1, padding=0):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return torch.relu(self.bn(self.conv(x)))

    class Conv1x1(ConvLayer):
        def __init__(self, cin, cout):
            super().__init__(cin, cout, 1)

    class Conv1x1Linear(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, bias=False)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return self.bn(self.conv(x))

    class LightConv3x3(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
            self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return torch.relu(self.bn(self.conv2(self.conv1(x))))

    class ChannelGate(nn.Module):
        def __init__(self, c, reduction=16):
            super().__init__()
            self.fc1 = nn.Conv2d(c, c // reduction, 1, bias=True)
            self.fc2 = nn.Conv2d(c // reduction, c, 1, bias=True)

        def forward(self, x):
            g = x.mean(dim=(2, 3), keepdim=True)
            g = torch.sigmoid(self.fc2(torch.relu(self.fc1(g))))
            return x * g

    class OSBlock(nn.Module):
        def __init__(self, cin, cout, reduction=4):
            super().__init__()
            mid = cout // reduction
            self.conv1 = Conv1x1(cin, mid)
            self.conv2a = LightConv3x3(mid, mid)
            self.conv2b = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(2)])
            self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
            self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
            self.gate = ChannelGate(mid)
            self.conv3 = Conv1x1Linear(mid, cout)
            self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None

        def forward(self, x):
            x1 = self.conv1(x)
            x2 = (
                self.gate(self.conv2a(x1))
                + self.gate(self.conv2b(x1))
                + self.gate(self.conv2c(x1))
                + self.gate(self.conv2d(x1))
            )
            x3 = self.conv3(x2)
            identity = self.downsample(x) if self.downsample is not None else x
            return torch.relu(x3 + identity)

    class OSNet(nn.Module):
        def __init__(self, channels=(64, 256, 384, 512)):
            super().__init__()
            c = channels
            self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            # Stages 2 and 3 end with a transition (1x1 conv + 2x2 avg pool).
            self.conv2 = nn.Sequential(
                OSBlock(c[0], c[1]), OSBlock(c[1], c[1]),
                nn.Sequential(Conv1x1(c[1], c[1]), nn.AvgPool2d(2, stride=2)),
            )
            self.conv3 = nn.Sequential(
                OSBlock(c[1], c[2]), OSBlock(c[2], c[2]),
                nn.Sequential(Conv1x1(c[2], c[2]), nn.AvgPool2d(2, stride=2)),
            )
            self.conv4 = nn.Sequential(OSBlock(c[2], c[3]), OSBlock(c[3], c[3]))
            self.conv5 = Conv1x1(c[3], c[3])
            self.fc = nn.Sequential(nn.Linear(c[3], 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))

        def forward(self, x):
            x = self.maxpool(self.conv1(x))
            x = self.conv3(self.conv2(x))
            x = self.conv5(self.conv4(x))
            return self.fc(x.mean(dim=(2, 3)))

    return OSNet(channels)


def _weights_path(variant: str) -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / f"osnet_{variant}_msmt17.pt"


def load_model(variant: str | None = None):
    """Build OSNet and load the MSMT17 weights (downloading once)."""
    global _model
    variant = variant or VARIANT
    if _model is not None and _model[0] == variant:
        return _model[1]
    import torch

    channels, url = VARIANTS[variant]
    path = _weights_path(variant)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading OSNet ReID weights to %s", path)
        import gdown  # Google Drive host used by the published torchreid weights

        gdown.download(url, str(path), quiet=True)
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    state = {k.removeprefix("module."): v for k, v in state.items() if not k.startswith("classifier.")}
    model = _build(channels)
    model.load_state_dict(state, strict=True)
    model.eval()
    _model = (variant, model)
    return model


def embed(frame_bgr, boxes: list[tuple[int, int, int, int]]):
    """L2-normalized 512-d embeddings, one per box (N x 512 numpy array)."""
    import cv2
    import numpy as np
    import torch

    if not boxes:
        return np.zeros((0, 512), np.float32)
    model = load_model()
    h_img, w_img = frame_bgr.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    batch = []
    for x, y, w, h in boxes:
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x + w), w_img), min(int(y + h), h_img)
        crop = frame_bgr[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]
        crop = cv2.resize(crop, (_INPUT_HW[1], _INPUT_HW[0]))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch.append(((rgb - mean) / std).transpose(2, 0, 1))
    with torch.no_grad():
        feats = model(torch.from_numpy(np.stack(batch)))
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats.numpy()
