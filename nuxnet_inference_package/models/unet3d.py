"""3D U-Net used for nucleus-marker segmentation.

This module is the **inference-side mirror** of the architecture in
``nuxnet-training``:

    numorph_nuclei_segmentation/model/unet_3d_models.py

The two files must stay structurally identical, because a trained checkpoint is
just a dictionary of tensors keyed by module path. Any difference in naming
(``inc.conv_block_1.conv_1`` vs ``inc.conv_1``), in connectivity (residual
shortcuts), or in layer configuration (``bias=False``) makes
``load_state_dict`` reject the checkpoint.

The architecture is a residual 3D U-Net:

* every ``ConvBlock`` has a skip connection, realised as ``nn.Identity`` when the
  channel counts match and as a 1x1 convolution + batch norm otherwise;
* convolutions carry no bias, because the following batch norm already has one;
* two 2x2 stride-2 convolutions downsample, nearest-neighbour upsampling joins the
  two encoder feature maps.

``tests/test_unet3d_parity.py`` pins this by comparing the parameter names and
shapes against a recorded golden file produced by the trainer, so a future edit on
either side fails loudly instead of silently producing an unusable checkpoint.

Reference commit of the training repository this mirrors:
``luiskuhn/nuxnet-training`` @ f716565.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["UNet3D"]

#: Default dropout, matching the trainer's ``--dropout-rate`` default.
DEFAULT_DROPOUT = 0.10


class ConvBlock(nn.Module):
    """Two-convolution residual block with spatial feature-map dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.dropout_1 = nn.Dropout3d(dropout)
        self.dropout_2 = nn.Dropout3d(dropout)

        # bias=False: the following BatchNorm3d supplies its own affine shift.
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm_1 = nn.BatchNorm3d(out_channels)

        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm3d(out_channels)

        self.non_linearity = nn.ReLU(inplace=False)
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.non_linearity(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.dropout_2(x)
        return self.non_linearity(x + identity)


class InputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block_1(x)


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = DEFAULT_DROPOUT) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=2, stride=2),
            nn.Dropout3d(dropout),
            ConvBlock(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSamplingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Dropout3d(dropout),
        )
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_1(x)


class UNet3D(nn.Module):
    """Residual 3D U-Net as trained by ``nuxnet-training``.

    Args:
        in_channels: input channels. The NuMorph nucleus-marker task uses 1.
        classes: output classes including background. The task uses 2
            (0 = background, 1 = nucleus).
        dropout: dropout probability applied inside every conv block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        classes: int = 2,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.inc = InputBlock(in_channels, 32, dropout=dropout)
        self.down1 = DownSamplingBlock(32, 64, dropout=dropout)
        self.down2 = DownSamplingBlock(64, 128, dropout=dropout)
        self.mid = ConvBlock(128, 128, dropout=dropout)
        self.up1 = UpSamplingBlock(128, 64, 64, dropout=dropout)
        self.up2 = UpSamplingBlock(64, 32, 32, dropout=dropout)
        self.outc = OutputBlock(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.mid(x3)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)
