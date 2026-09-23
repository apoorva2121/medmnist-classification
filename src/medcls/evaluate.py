"""Accuracy, macro AUC, expected calibration error, temperature scaling, Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict_logits(model, loader):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x)); ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def accuracy(logits, y) -> float:
    return float((logits.argmax(1) == y).float().mean())


def macro_auc(logits, y, n_classes: int) -> float:
    p = F.softmax(logits, dim=1).numpy()
    yn = y.numpy()
    if n_classes == 2:
        return float(roc_auc_score(yn, p[:, 1]))
    return float(roc_auc_score(yn, p, multi_class="ovr", average="macro"))


def ece(logits, y, n_bins: int) -> float:
    """Expected calibration error on the max-probability prediction."""
    p = F.softmax(logits, dim=1)
    conf, pred = p.max(1)
    correct = (pred == y).float()
    bins = torch.linspace(0, 1, n_bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    return float(e)


def fit_temperature(logits, y) -> float:
    """Single scalar T minimising NLL on the validation split."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / logT.exp(), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def grad_cam(model, x: torch.Tensor, target_class: int | None = None) -> np.ndarray:
    """Grad-CAM on the last conv feature map for one image (1,C,H,W). Returns HxW in [0,1]."""
    model.eval()
    feats = {}
    def hook(_, __, out):
        out.retain_grad(); feats["a"] = out
    h = model.features.register_forward_hook(hook)
    logits = model(x)
    cls = int(logits.argmax(1)) if target_class is None else target_class
    logits[0, cls].backward()
    h.remove()
    a, g = feats["a"], feats["a"].grad
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().numpy()
