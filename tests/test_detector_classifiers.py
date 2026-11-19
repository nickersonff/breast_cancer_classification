import pytest
from torch import randn

from pt.models.model_factory import ModelFactory


@pytest.mark.parametrize("architecture", ["retinanet", "faster-rcnn", "yolo"])
def test_detector_architectures_return_classification_logits(
    architecture: str,
) -> None:
    model = ModelFactory.create_model(architecture, num_classes=2)
    logits = model(randn(2, 3, 64, 64))

    assert tuple(logits.shape) == (2, 2)
