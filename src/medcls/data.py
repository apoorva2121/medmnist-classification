"""MedMNIST loaders with train-only augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import medmnist
import numpy as np
import torch
import yaml
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class MedDataset(Dataset):
    def __init__(self, flag: str, split: str, root: str, augment: bool, mean: float, std: float) -> None:
        cls = getattr(medmnist, INFO[flag]["python_class"])
        ds = cls(split=split, download=True, root=root)
        x = ds.imgs.astype("float32") / 255.0
        if x.ndim == 3:
            x = x[..., None]
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2)         # N,C,28,28
        self.x = (self.x - mean) / std
        self.y = torch.from_numpy(ds.labels.astype("int64")).squeeze(1)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[2])
            pad = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")
            dx, dy = torch.randint(0, 5, (2,))
            x = pad[:, dy:dy + 28, dx:dx + 28]
        return x, self.y[i]


def make_loaders(flag: str, cfg: dict[str, Any]):
    info = INFO[flag]
    root = cfg["paths"]["data"]
    # normalisation statistics from the training split only
    cls = getattr(medmnist, info["python_class"])
    tr = cls(split="train", download=True, root=root).imgs.astype("float32") / 255.0
    mean, std = float(tr.mean()), float(tr.std())
    t = cfg["training"]
    mk = lambda split, aug: DataLoader(MedDataset(flag, split, root, aug, mean, std),
                                       batch_size=t["batch_size"], shuffle=(split == "train"),
                                       num_workers=t["num_workers"])
    n_classes = len(info["label"])
    in_ch = 3 if info["n_channels"] == 3 else 1
    return mk("train", t["augment"]), mk("val", False), mk("test", False), n_classes, in_ch, info["task"]
