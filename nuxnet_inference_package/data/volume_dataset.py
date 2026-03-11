from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VolumeDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for 3D input/label pairs.

    Supports legacy pairs `img_<id>.pt` + `lab_<id>.pt` and `.npy` pairs.
    """

    def __init__(self, ids: list[int], path: str = "data/") -> None:
        self.inputs: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        base = Path(path)

        for sample in ids:
            img_pt = base / f"img_{sample}.pt"
            lab_pt = base / f"lab_{sample}.pt"
            img_npy = base / f"img_{sample}.npy"
            lab_npy = base / f"lab_{sample}.npy"

            if img_pt.exists() and lab_pt.exists():
                vol = torch.load(img_pt, map_location="cpu")
                label = torch.load(lab_pt, map_location="cpu")
                vol_np = np.asarray(vol, dtype=np.float32)
                label_np = np.asarray(label, dtype=np.float32)
            elif img_npy.exists() and lab_npy.exists():
                vol_np = np.load(img_npy).astype(np.float32)
                label_np = np.load(lab_npy).astype(np.float32)
            else:
                raise FileNotFoundError(f"Could not find img/lab pair for id {sample} in {base}")

            vol_np = np.expand_dims(vol_np, axis=0)
            self.inputs.append(torch.tensor(vol_np, dtype=torch.float32))
            self.labels.append(torch.tensor(label_np, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]
