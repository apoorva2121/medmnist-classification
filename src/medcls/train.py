"""Train each model on each dataset, evaluate on the official test split, calibrate, save Grad-CAMs.

Usage:
    python -m medcls.train [--config config.yaml] [--datasets pneumoniamnist] [--models small_cnn] [--epochs N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_config, make_loaders
from .evaluate import accuracy, ece, fit_temperature, grad_cam, macro_auc, predict_logits
from .models import make_model


def train_one(model, train_loader, val_loader, cfg, log):
    t = cfg["training"]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    best, best_state, bad = -1.0, None, 0
    for ep in range(t["epochs"]):
        model.train(); tl = 0.0; n = 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(y); n += len(y)
        sched.step()
        vl, vy = predict_logits(model, val_loader)
        vacc = accuracy(vl, vy)
        log(f"    epoch {ep + 1:02d}  train_loss {tl / n:.4f}  val_acc {vacc:.4f}")
        if vacc > best:
            best, best_state, bad = vacc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= t["early_stopping_patience"]:
                log("    early stop"); break
    model.load_state_dict(best_state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--datasets", nargs="*"); ap.add_argument("--models", nargs="*")
    ap.add_argument("--epochs", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs: cfg["training"]["epochs"] = args.epochs
    torch.manual_seed(cfg["training"]["seed"]); np.random.seed(cfg["training"]["seed"])
    rep = Path(cfg["paths"]["reports"]); rep.mkdir(parents=True, exist_ok=True)
    art = Path(cfg["paths"]["artifacts"]); art.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or list(cfg["datasets"])
    models = args.models or cfg["models"]
    results = []
    t0 = time.time()
    for flag in datasets:
        tr, va, te, n_classes, in_ch, task = make_loaders(flag, cfg)
        print(f"\n[{flag}] {task} | {n_classes} classes | in_ch {in_ch} | train {len(tr.dataset)} val {len(va.dataset)} test {len(te.dataset)}")
        for mname in models:
            t = time.time()
            model = make_model(mname, in_ch, n_classes)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {mname} ({n_params/1e6:.2f}M params)")
            model = train_one(model, tr, va, cfg, print)
            vl, vy = predict_logits(model, va)
            tl, ty = predict_logits(model, te)
            T = fit_temperature(vl, vy) if cfg["calibration"]["temperature_scaling"] else 1.0
            nb = cfg["calibration"]["n_bins"]
            row = {"dataset": flag, "model": mname, "params_M": round(n_params / 1e6, 2),
                   "test_acc": round(accuracy(tl, ty), 4), "test_auc": round(macro_auc(tl, ty, n_classes), 4),
                   "test_ece_raw": round(ece(tl, ty, nb), 4), "test_ece_temp_scaled": round(ece(tl / T, ty, nb), 4),
                   "temperature": round(T, 3), "train_seconds": round(time.time() - t, 1),
                   **cfg["datasets"][flag]}
            results.append(row)
            print(f"    TEST acc {row['test_acc']:.4f}  auc {row['test_auc']:.4f}  "
                  f"ece {row['test_ece_raw']:.4f} -> {row['test_ece_temp_scaled']:.4f} (T={T:.2f})  {row['train_seconds']}s")
            torch.save(model.state_dict(), art / f"{flag}_{mname}.pt")
            # Grad-CAM for the first 8 test images
            xs = torch.stack([te.dataset[i][0] for i in range(8)])
            cams = np.stack([grad_cam(model, xs[i:i + 1]) for i in range(8)])
            np.save(rep / f"gradcam_{flag}_{mname}.npy", cams)

    import pandas as pd
    df = pd.DataFrame(results)
    print("\n", df.to_string(index=False))
    df.to_csv(rep / "results.csv", index=False)
    (rep / "results.json").write_text(json.dumps({"results": results, "runtime_seconds": round(time.time() - t0, 1),
                                                   "config": cfg}, indent=2, default=str))


if __name__ == "__main__":
    main()
