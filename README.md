# MedMNIST classification on a CPU laptop — small CNN vs ResNet-18, with calibration and Grad-CAM

I wanted a deep-learning project that is honest about hardware. Most image-classification repos assume a
GPU; this one trains two architectures on two MedMNIST v2 datasets on a CPU-only laptop, compares them
against the published ResNet-18 numbers from the MedMNIST paper, and adds the two things I always want
from a medical classifier and rarely see in a benchmark: **how calibrated are the probabilities**, and
**where is the model looking**.

MedMNIST v2 (Yang et al., 2023) is a collection of 28×28 biomedical image datasets with fixed
train/val/test splits, so results are directly comparable to the paper. I used:

| dataset | task | classes | train / val / test | modality |
|---|---|---|---|---|
| PneumoniaMNIST | binary | 2 | 4,708 / 524 / 624 | paediatric chest X-ray |
| DermaMNIST | multi-class | 7 | 7,007 / 1,003 / 2,005 | dermatoscope (HAM10000) |

## Workflow

```mermaid
flowchart LR
    D[medmnist download<br/>official splits] --> N[normalise with<br/>train-split mean/std]
    N --> A[augment train only<br/>random crop + h-flip]
    A --> T[train 12 epochs, AdamW<br/>early stop on val acc]
    T --> S[select best val epoch]
    S --> C[fit temperature T<br/>on validation logits]
    C --> E[evaluate ONCE on test<br/>acc · macro AUC · ECE raw vs T-scaled]
    E --> G[Grad-CAM on 8 test images]
```

Two models, both defined in `src/medcls/models.py`:

- **small_cnn** — three conv blocks (32→64→128), BatchNorm, global average pooling, dropout 0.3. 0.29 M
  parameters. Trains in 3–4 minutes per dataset on CPU.
- **resnet18_28** — the ResNet-18 topology with a 3×3 stem and no initial max-pool, which is the variant
  MedMNIST uses for 28 px inputs. 11.17 M parameters. 22–43 minutes per dataset on CPU.

Training: AdamW, lr 1e-3, weight decay 1e-4, batch 128, 12 epochs, early stopping with patience 4 on
validation accuracy, best-epoch weights restored. Seed 42. The test split is touched exactly once, after
the model and the temperature are frozen.

## Results (official test splits)

From `reports/results.csv`; "published" is the ResNet-18 @28 px row in the MedMNIST v2 paper.

| dataset | model | params | test acc | test AUC | published acc | published AUC | ECE raw → T-scaled (T) | train time |
|---|---|---|---|---|---|---|---|---|
| PneumoniaMNIST | small_cnn | 0.29 M | **0.886** | **0.979** | 0.854 | 0.944 | 0.074 → 0.075 (0.91) | 4.0 min |
| PneumoniaMNIST | resnet18_28 | 11.17 M | 0.862 | 0.975 | 0.854 | 0.944 | 0.089 → 0.088 (1.03) | 22 min |
| DermaMNIST | small_cnn | 0.29 M | **0.779** | **0.933** | 0.735 | 0.917 | 0.028 → 0.027 (0.98) | 2.9 min |
| DermaMNIST | resnet18_28 | 11.17 M | 0.765 | 0.928 | 0.735 | 0.917 | 0.023 → 0.023 (0.99) | 43 min |

### What I take from this

**The 0.29 M-parameter CNN beats the published ResNet-18 on both datasets, and beats my own ResNet-18
too.** On PneumoniaMNIST it is +3.2 points accuracy and +3.5 points AUC over the paper; on DermaMNIST
+4.4 points accuracy and +1.6 AUC. I do not think this says small CNNs are better than ResNets. It says
that at 28×28 with a few thousand training images, 11 M parameters is far more capacity than the data
can constrain in 12 CPU epochs — ResNet-18 was still improving on validation when the epoch budget ran
out (DermaMNIST val acc 0.70 → 0.78 over 12 epochs), while the small CNN had converged. With a GPU and
50+ epochs I would expect the ordering to flip; with this compute budget the small model is the right
choice, and that is the kind of trade-off I want a benchmark to make visible.

The published numbers are also from a stronger schedule (100 epochs, lr decay), so the comparison is
"my CPU budget vs their GPU budget", not architecture vs architecture. I include them as a sanity floor,
not as a claim of a new state of the art.

**Calibration.** ECE is already low on DermaMNIST (0.02–0.03) and temperature scaling changes almost
nothing (T ≈ 1). On PneumoniaMNIST the ECE is higher (0.07–0.09) and temperature scaling does *not*
fix it — the small CNN gets marginally worse. With only 524 validation images the fitted T is noisy,
and the miscalibration is not a uniform over/under-confidence that a single scalar can correct. I
report the before/after numbers rather than drop the technique, because "temperature scaling didn't
help here, and here is why" is more useful than a table where it always helps.

**Grad-CAM.** `reports/gradcam_*.npy` holds 8 test-image heatmaps per model. On the chest X-rays the
small CNN's activation concentrates in the lung fields; the ResNet's maps are more diffuse, consistent
with an under-trained larger network. The maps are saved as arrays so they can be inspected without a
plotting dependency.

## Leakage and evaluation hygiene

- Normalisation statistics are computed on the **training split only** (`data.py`).
- Augmentation is applied to the training loader only.
- Model selection (best epoch) and temperature fitting use the **validation split only**.
- The test split is evaluated once per model, after everything is frozen.
- Splits are MedMNIST's official ones, so no images cross between them.

## Repository layout

```
config.yaml               datasets, published reference numbers, training and calibration settings
src/medcls/
  data.py                 MedMNIST loaders, train-only normalisation, augmentation
  models.py               SmallCNN, ResNet18_28
  evaluate.py             accuracy, macro AUC, ECE, temperature scaling, Grad-CAM
  train.py                the full loop; writes reports/results.{csv,json} and Grad-CAM arrays
tests/test_models.py      shape / parameter-count / metric sanity tests
reports/                  results and Grad-CAM heatmaps from the run above
```

## Reproduce

```bash
pip install -e .
KMP_DUPLICATE_LIB_OK=TRUE python -m medcls.train      # ~75 min on a CPU laptop, downloads ~30 MB
pytest -q
```

Set `training.epochs: 2` in `config.yaml` for a 3-minute smoke run. The datasets download automatically
via the `medmnist` package.

## What I would do next

- Run ResNet-18 to convergence on a GPU to confirm the capacity story above, and add a cosine schedule.
- Try the 64 px and 128 px MedMNIST+ variants where the ResNet should have more to work on.
- Replace scalar temperature scaling with class-wise or Dirichlet calibration for the multi-class case.
- Add a proper Grad-CAM figure once I settle on a plotting dependency.

## References

- Yang, J. et al. (2023). *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical
  image classification.* Scientific Data.
- Guo, C. et al. (2017). *On calibration of modern neural networks.* ICML.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual explanations from deep networks.* ICCV.
