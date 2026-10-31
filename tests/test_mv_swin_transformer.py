import torch

from pt.models.model_factory import ModelFactory
from pt.models.mv_swin_transformer import MVSwinTransformer


def test_mv_swin_accepts_single_and_paired_views() -> None:
    model = MVSwinTransformer(num_classes=2).eval()
    with torch.no_grad():
        single = model(torch.randn(1, 1, 224, 224))
        paired = model(torch.randn(1, 2, 1, 224, 224))
    assert single.shape == (1, 2)
    assert paired.shape == (1, 2)


def test_factory_registers_mv_swin_t() -> None:
    assert isinstance(ModelFactory.create_model("mv_swin_t", 2), MVSwinTransformer)
