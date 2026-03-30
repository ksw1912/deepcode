import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from dataloaders.data_loader_fakeV import Fake_Vaihingen_LoveDA
from torch.utils.data import DataLoader, random_split
# ============================================================
# Spectral preprocessing
# Based on the paper's RGBVI, MGRVI, and Laplace filtering.
# Input RGB image tensor is expected in range [0, 1].
# ============================================================


def _safe_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return num / (den + eps)


class SpectralPreprocessor(nn.Module):
    """
    Build 2-channel spectral information from RGB image.

    RGBVI = (G^2 - B*R) / (G^2 + B*R)
    MGRVI = (G^2 - R^2) / (G^2 + R^2)

    Then apply Laplace filtering to each index channel.
    The paper uses vegetation indices + Laplace filtering as the spectral
    information branch input.
    """

    def __init__(self):
        super().__init__()
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", kernel)

    def laplace(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        weight = self.laplace_kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, weight, padding=1, groups=c)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B, 3, H, W], range [0, 1]
        r = rgb[:, 0:1]
        g = rgb[:, 1:2]
        b = rgb[:, 2:3]

        rgbvi = _safe_div(g.pow(2) - b * r, g.pow(2) + b * r)
        mgrvi = _safe_div(g.pow(2) - r.pow(2), g.pow(2) + r.pow(2))

        si = torch.cat([self.laplace(rgbvi), self.laplace(mgrvi)], dim=1)
        return si


# ============================================================
# Common building blocks
# ============================================================


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
    ):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class SpatialAttention(nn.Module):
    """CBAM-style SAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_map, max_map], dim=1))
        return self.sigmoid(attn)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return w


class MLP2d(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# GLTB: simplified practical implementation
# Paper describes a final encoder block that mixes local and global context.
# This implementation keeps the intent while remaining trainable and concise.
# ============================================================


class WindowSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, window_size: int = 8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x, pad_h, pad_w = self._pad(x)
        _, _, hp, wp = x.shape
        ws = self.window_size

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        def window_partition(t: torch.Tensor) -> torch.Tensor:
            t = t.view(b, self.num_heads, self.head_dim, hp, wp)
            t = t.view(b, self.num_heads, self.head_dim, hp // ws, ws, wp // ws, ws)
            t = t.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
            t = t.view(-1, self.num_heads, ws * ws, self.head_dim)
            return t

        qw = window_partition(q)
        kw = window_partition(k)
        vw = window_partition(v)

        attn = torch.matmul(qw, kw.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, vw)

        out = out.view(b, hp // ws, wp // ws, self.num_heads, ws, ws, self.head_dim)
        out = out.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
        out = out.view(b, c, hp, wp)
        out = self.proj(out)

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


class EfficientGlobalLocalAttention(nn.Module):
    """
    Paper description:
    - global branch: windowed self-attention + horizontal/vertical pooling exchange
    - local branch: two parallel convolutions with different kernel sizes

    This is a faithful engineering approximation.
    """

    def __init__(self, channels: int, num_heads: int = 8, window_size: int = 8):
        super().__init__()
        self.global_attn = WindowSelfAttention(channels, num_heads=num_heads, window_size=window_size)
        self.h_pool_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.w_pool_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.local_3 = ConvBNReLU(channels, channels, kernel_size=3)
        self.local_5 = ConvBNReLU(channels, channels, kernel_size=5)
        self.local_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_ctx = self.global_attn(x)
        # exchange between windows approximation via axis pooled context
        h_ctx = self.h_pool_proj(F.adaptive_avg_pool2d(x, (x.shape[-2], 1)).expand_as(x))
        w_ctx = self.w_pool_proj(F.adaptive_avg_pool2d(x, (1, x.shape[-1])).expand_as(x))
        global_ctx = global_ctx + h_ctx + w_ctx

        local_ctx = self.local_fuse(torch.cat([self.local_3(x), self.local_5(x)], dim=1))
        return global_ctx + local_ctx


class GLTB(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, window_size: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = EfficientGlobalLocalAttention(channels, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.BatchNorm2d(channels)
        self.mlp = MLP2d(channels, expansion=mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.attn(self.norm1(x))
        y = y + self.mlp(self.norm2(y))
        return y


# ============================================================
# SSIAB
# ============================================================


class SSIAB(nn.Module):
    """
    Spectral Semantic Information Aggregation Block.

    Paper intent:
    1) extract directional features with 1x5 and 5x1 convs from Fi and Si
    2) concatenate them and build spatial attention weight SAW
    3) enhance Fi and Si with SAW and weighted-sum fuse
    """

    def __init__(self, channels: int):
        super().__init__()
        self.f_h = ConvBNReLU(channels, channels, kernel_size=(1, 5), padding=(0, 2))
        self.f_v = ConvBNReLU(channels, channels, kernel_size=(5, 1), padding=(2, 0))
        self.s_h = ConvBNReLU(channels, channels, kernel_size=(1, 5), padding=(0, 2))
        self.s_v = ConvBNReLU(channels, channels, kernel_size=(5, 1), padding=(2, 0))

        self.sam = SpatialAttention(kernel_size=7)
        self.weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, kernel_size=1, bias=True),
        )

    def forward(self, f: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        f_dir = self.f_h(f) + self.f_v(f)
        s_dir = self.s_h(s) + self.s_v(s)

        saw = self.sam(torch.cat([f_dir, s_dir], dim=1))
        f_enh = f * saw
        s_enh = s * saw

        alpha = self.weight_gen(torch.cat([f_enh, s_enh], dim=1))
        alpha = torch.softmax(alpha, dim=1)
        sf = alpha[:, 0:1] * f_enh + alpha[:, 1:2] * s_enh
        return sf


# ============================================================
# FDB and ODB
# ============================================================


class FDB(nn.Module):
    """
    Fusion Decoder Block.

    Paper description:
    - upsample higher-level FF_{i+1}
    - concat with same-scale SF_i
    - channel attention weight
    - weighted residual enhancement
    - 1x1 conv + BN + ReLU for channel reduction/integration
    - 3x3 conv + BN + ReLU for output FF_i
    """

    def __init__(self, skip_ch: int, high_ch: int, out_ch: int):
        super().__init__()
        in_ch = skip_ch + high_ch
        self.ca = ChannelAttention(in_ch)
        self.reduce = ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0)
        self.refine = ConvBNReLU(out_ch, out_ch, kernel_size=3)

    def forward(self, sf_i: torch.Tensor, ff_next: torch.Tensor) -> torch.Tensor:
        ff_next = F.interpolate(ff_next, size=sf_i.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([sf_i, ff_next], dim=1)
        w = self.ca(x)
        x = x * w + x
        x = self.reduce(x)
        x = self.refine(x)
        return x


class ODB(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int = 64, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch, kernel_size=3),
            nn.Conv2d(mid_ch, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return self.conv(x)


# ============================================================
# ResNet18 backbone wrappers
# ============================================================


class ResNet18Encoder(nn.Module):
    """
    Returns 4 multi-scale feature maps: F1, F2, F3, F4.

    Following the paper:
    - use ResNet18 as backbone
    - last stage replaced by GLTB

    Here:
    F1 = layer1 output (1/4)
    F2 = layer2 output (1/8)
    F3 = layer3 output (1/16)
    F4 = GLTB(F3)   (1/16)

    This preserves the paper's statement of 3 ResBlocks + 1 GLTB.
    """

    def __init__(self, in_channels: int, pretrained: bool = False):
        super().__init__()
        base = resnet18(weights=None if not pretrained else "DEFAULT")

        # Replace first conv if needed.
        if in_channels != 3:
            old = base.conv1
            base.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    base.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # 64, 1/4
        self.layer2 = base.layer2  # 128, 1/8
        self.layer3 = base.layer3  # 256, 1/16
        self.gltb = GLTB(256, num_heads=8, window_size=8, mlp_ratio=4)

        # Normalize channels to decoder-friendly widths.
        self.proj1 = nn.Conv2d(64, 64, kernel_size=1)
        self.proj2 = nn.Conv2d(128, 128, kernel_size=1)
        self.proj3 = nn.Conv2d(256, 256, kernel_size=1)
        self.proj4 = nn.Conv2d(256, 256, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f1 = self.proj1(self.layer1(x))
        f2 = self.proj2(self.layer2(f1))
        f3 = self.proj3(self.layer3(f2))
        f4 = self.proj4(self.gltb(f3))
        return f1, f2, f3, f4


# ============================================================
# SIGNet
# ============================================================


class SIGNet(nn.Module):
    def __init__(self, rgb_pretrained: bool = True, si_pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        self.spectral = SpectralPreprocessor()

        self.rgb_encoder = ResNet18Encoder(in_channels=3, pretrained=rgb_pretrained)
        self.si_stem = nn.Sequential(
            ConvBNReLU(2, 32, kernel_size=3),
            ConvBNReLU(32, 64, kernel_size=3, stride=2),
        )
        self.si_encoder = ResNet18Encoder(in_channels=64, pretrained=si_pretrained)

        self.ssiab1 = SSIAB(64)
        self.ssiab2 = SSIAB(128)
        self.ssiab3 = SSIAB(256)
        self.ssiab4 = SSIAB(256)

        self.ff4_proj = ConvBNReLU(256, 256, kernel_size=3)
        self.fdb3 = FDB(skip_ch=256, high_ch=256, out_ch=256)
        self.fdb2 = FDB(skip_ch=128, high_ch=256, out_ch=128)
        self.fdb1 = FDB(skip_ch=64, high_ch=128, out_ch=64)
        self.odb = ODB(64, mid_ch=64, num_classes=num_classes)

    def forward(self, rgb: torch.Tensor) -> Dict[str, torch.Tensor]:
        out_size = rgb.shape[-2:]

        # RGB branch
        f1, f2, f3, f4 = self.rgb_encoder(rgb)

        # SI branch
        si = self.spectral(rgb)
        # Bring 2-channel SI to a richer shallow embedding before the ResNet-style encoder.
        si0 = self.si_stem(si)
        s1, s2, s3, s4 = self.si_encoder(si0)

        # Align SI scales to RGB scales if needed.
        if s1.shape[-2:] != f1.shape[-2:]:
            s1 = F.interpolate(s1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        if s2.shape[-2:] != f2.shape[-2:]:
            s2 = F.interpolate(s2, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        if s3.shape[-2:] != f3.shape[-2:]:
            s3 = F.interpolate(s3, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        if s4.shape[-2:] != f4.shape[-2:]:
            s4 = F.interpolate(s4, size=f4.shape[-2:], mode="bilinear", align_corners=False)

        # Same-scale semantic/spectral fusion
        sf1 = self.ssiab1(f1, s1)
        sf2 = self.ssiab2(f2, s2)
        sf3 = self.ssiab3(f3, s3)
        sf4 = self.ssiab4(f4, s4)

        # Decoder
        ff4 = self.ff4_proj(sf4)
        ff3 = self.fdb3(sf3, ff4)
        ff2 = self.fdb2(sf2, ff3)
        ff1 = self.fdb1(sf1, ff2)
        logits = self.odb(ff1, out_size=out_size)

        return {
            "logits": logits,
            "mask_prob": torch.sigmoid(logits),
            "spectral_info": si,
        }


# ============================================================
# Loss and training helpers
# ============================================================



    """
    The paper uses weighted cross-entropy with tampered class weighted 20x.
    For binary segmentation with logits, BCEWithLogits is a practical match.
    """

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 20.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            pos_weight=self.pos_weight
        )


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 100
    pos_weight: float = 20.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SIGNet PyTorch implementation")
    parser.add_argument("--mode", type=str, default="train_step", choices=["sanity", "train_step"], help="Run mode")
    parser.add_argument("--batch-size", type=int, default=2, help="Input batch size for sanity or dummy train step")
    parser.add_argument("--height", type=int, default=512, help="Input image height")
    parser.add_argument("--width", type=int, default=512, help="Input image width")
    parser.add_argument("--num-classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--rgb-pretrained", action="store_true", help="Use pretrained weights for RGB encoder")
    parser.add_argument("--si-pretrained", action="store_true", help="Use pretrained weights for SI encoder")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive class weight for BCE loss")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--shuffle", type=bool, default=True, help="disable data shuffling")
    return parser


# ============================================================
# Example training step
# ============================================================


def train_one_step(model: nn.Module, train_loader, optimizer: torch.optim.Optimizer, criterion: nn.Module,args):
    device = args.device
    for batch in train_loader:
        model.train()
        images = batch["image"].to(device)
        masks = batch["mask"].to(device).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss = criterion(out["logits"], masks)
        print(out["logits"].shape    )
        loss.backward()
        optimizer.step()
        break
    return {"loss": float(loss.detach().cpu())}


# ============================================================
# Dummy sanity check
# ============================================================


def _sanity_check(args):
    device = args.device
    model = SIGNet(
        rgb_pretrained=args.rgb_pretrained,
    ).to(device)
    x = torch.rand(args.batch_size, 3, args.height, args.width, device=device)
    y = model(x)
    print("logits:", y["logits"].shape)
    print("mask_prob:", y["mask_prob"].shape)
    print("spectral_info:", y["spectral_info"].shape)


def _dummy_train_step(args):
    device = args.device
    model = SIGNet(
        rgb_pretrained=args.rgb_pretrained
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = WeightedBCELoss(pos_weight=args.pos_weight).to(device)

    train_dataset_not_split = Fake_Vaihingen_LoveDA(root_dir='../../dataset/Fake-Vaihingen', split="train")


    train_size = int(0.8 * len(train_dataset_not_split))
    val_size = len(train_dataset_not_split) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset_not_split,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, drop_last=True)

    log = train_one_step(model, train_loader, optimizer, criterion,args)
    print("train_step_loss:", log["loss"])


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "sanity":
        _sanity_check(args)
    elif args.mode == "train_step":
        _dummy_train_step(args)
