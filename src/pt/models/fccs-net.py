num_classes=2
import torch
import torch.nn as nn
import torchvision.models as models

"""
    fonte: https://www.sciencedirect.com/science/article/abs/pii/S1746809424003161
"""

# Define the CBAM module
class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            #nn.MaxPool2d(1),
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_channel_att = self.channel_att(x)
        x_spatial_att = self.spatial_att(x)
        x_att = torch.mul(x_channel_att, x_spatial_att)
        return torch.mul(x, x_att)

#CBAM with residual connection
class ResNet18_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18_CBAM, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

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

        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(960, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_res1 = self.resnet.layer1(x)
        x_cbam1 = self.cbam1(x_res1)
        x1 = x_res1 + x_cbam1

        x_res2 = self.resnet.layer2(x1)
        x_cbam2 = self.cbam2(x_res2)
        x2 = x_res2 + x_cbam2

        x_res3 = self.resnet.layer3(x2)
        x_cbam3 = self.cbam3(x_res3)
        x3 = x_res3 + x_cbam3

        x_res4 = self.resnet.layer4(x3)
        x_cbam4 = self.cbam4(x_res4)
        x4 = x_res4 + x_cbam4

        x_cbam1_gap = self.global_avg_pooling(x_cbam1)
        x_cbam2_gap = self.global_avg_pooling(x_cbam2)
        x_cbam3_gap = self.global_avg_pooling(x_cbam3)
        x_cbam4_gap = self.global_avg_pooling(x_cbam4)

        x_cbam_concat = torch.cat([x_cbam1_gap, x_cbam2_gap, x_cbam3_gap, x_cbam4_gap], dim=1)
        x_cbam_concat = x_cbam_concat.view(x_cbam_concat.size(0), -1)
        out = self.fc1(x_cbam_concat)
        out = self.fc2(out)
        return out


