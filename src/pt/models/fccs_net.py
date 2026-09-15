num_classes = 2
from torch import Tensor, cat, mul
from torch.nn import (
    AdaptiveAvgPool2d,
    Conv2d,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
)
from torchvision.models import ResNet, resnet18

"""
    fonte: https://www.sciencedirect.com/science/article/abs/pii/S1746809424003161
"""


# Define the CBAM module
class CBAM(Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_att = Sequential(
            AdaptiveAvgPool2d(1),
            Conv2d(in_channels, in_channels // 16, kernel_size=1),
            ReLU(inplace=True),
            Conv2d(in_channels // 16, in_channels, kernel_size=1),
            Sigmoid(),
        )
        self.spatial_att = Sequential(
            AdaptiveAvgPool2d(1),
            # MaxPool2d(1),
            Conv2d(in_channels, 1, kernel_size=7, padding=3),
            Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_channel_att: Tensor = self.channel_att(x)
        x_spatial_att: Tensor = self.spatial_att(x)
        x_att: Tensor = mul(x_channel_att, x_spatial_att)
        return mul(x, x_att)


# CBAM with residual connection
class ResNet18CBAM(Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet: ResNet = resnet18(pretrained=True)

        # Disable gradients for all the parameters in the pre-trained ResNet18
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Enable gradients for the last two layers of ResNet18
        for param in self.resnet.layer4.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        self.cbam1 = CBAM(64)  # Apply CBAM to the output of layer3
        self.cbam2 = CBAM(128)  # Apply CBAM to the output of layer4
        self.cbam3 = CBAM(256)  # Apply CBAM to the output of layer3
        self.cbam4 = CBAM(512)  # Apply CBAM to the output of layer4

        self.global_avg_pooling = AdaptiveAvgPool2d((1, 1))
        self.fc1 = Linear(960, 128)
        self.fc2 = Linear(128, num_classes)

    def forward(self, x: Tensor) -> Linear:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_res1: Sequential = self.resnet.layer1(x)
        x_cbam1: CBAM = self.cbam1(x_res1)
        x1: Sequential = x_res1 + x_cbam1

        x_res2: Sequential = self.resnet.layer2(x1)
        x_cbam2: CBAM = self.cbam2(x_res2)
        x2: Sequential = x_res2 + x_cbam2

        x_res3: Sequential = self.resnet.layer3(x2)
        x_cbam3: CBAM = self.cbam3(x_res3)
        x3: Sequential = x_res3 + x_cbam3

        x_res4: Sequential = self.resnet.layer4(x3)
        x_cbam4: CBAM = self.cbam4(x_res4)

        x_cbam1_gap: Tensor = self.global_avg_pooling(x_cbam1)
        x_cbam2_gap: Tensor = self.global_avg_pooling(x_cbam2)
        x_cbam3_gap: Tensor = self.global_avg_pooling(x_cbam3)
        x_cbam4_gap: Tensor = self.global_avg_pooling(x_cbam4)

        x_cbam_concat: Tensor = cat(
            [x_cbam1_gap, x_cbam2_gap, x_cbam3_gap, x_cbam4_gap], dim=1
        )
        x_cbam_concat = x_cbam_concat.view(x_cbam_concat.size(0), -1)
        out: Linear = self.fc1(x_cbam_concat)
        out: Linear = self.fc2(out)
        return out
