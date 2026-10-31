from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torchvision.models import ResNet, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

"""
Models for image classification using the feature extractors of popular object detection architectures.

https://www.sciencedirect.com/science/article/pii/S2589791825000192

"""

class RetinaNetClassifier(Module):
    """Image classifier using a RetinaNet-style FPN feature aggregation."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone: ResNet = resnet50(weights=None)
        self.features = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "c3",
                "layer3": "c4",
                "layer4": "c5",
            },
        )
        self.projections = Sequential(
            Conv2d(2048, 256, kernel_size=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
        )
        self.head = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Dropout(0.5),
            Linear(256, num_classes),
        )

    def forward(self, image: Tensor) -> Tensor:
        features = self.features(image)
        pyramid_feature = self.projections(features["c5"])
        return self.head(pyramid_feature)


class FasterRCNNClassifier(Module):
    """Image classifier using a Faster R-CNN ResNet-FPN-style backbone."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone: ResNet = resnet50(weights=None)
        self.features = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "p2",
                "layer3": "p3",
                "layer4": "p4",
            },
        )
        self.projections = Sequential(
            Conv2d(2048, 256, kernel_size=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
        )
        self.head = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Dropout(0.5),
            Linear(256, num_classes),
        )

    def forward(self, image: Tensor) -> Tensor:
        features = self.features(image)
        pooled_feature = self.projections(features["p4"])
        return self.head(pooled_feature)


class YOLOClassifier(Module):
    """YOLO-style convolutional classifier with no detection outputs."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = Sequential(
            Conv2d(3, 32, 3, stride=2, padding=1),
            BatchNorm2d(32),
            ReLU(inplace=True),
            Conv2d(32, 64, 3, stride=2, padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, 128, 3, stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 256, 3, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
        )
        self.head = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Dropout(0.5),
            Linear(256, num_classes),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.head(self.backbone(image))
