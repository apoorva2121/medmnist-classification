import torch

from medcls.evaluate import ece, fit_temperature
from medcls.models import make_model


def test_models_forward_shapes():
    for name in ["small_cnn", "resnet18_28"]:
        for in_ch, n in [(1, 2), (3, 7)]:
            m = make_model(name, in_ch, n)
            assert m(torch.zeros(4, in_ch, 28, 28)).shape == (4, n)


def test_ece_zero_when_perfectly_calibrated_and_confident():
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    y = torch.tensor([0, 1])
    assert ece(logits, y, 15) < 1e-3


def test_temperature_reduces_overconfidence():
    torch.manual_seed(0)
    y = torch.randint(0, 2, (500,))
    logits = torch.randn(500, 2) * 5  # overconfident random logits
    T = fit_temperature(logits, y)
    assert T > 1.0
