import torch.nn.functional as F
from safetensors.torch import load_model
from torch import Tensor
from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torchvision.models import (
    VGG,
    DenseNet,
    DenseNet121_Weights,
    ResNet,
    ResNet50_Weights,
    VGG16_BN_Weights,
    densenet121,
    resnet50,
    vgg16_bn,
)

from pt.utils.custom_fc import CustomFC


class MammographyEnsemble(Module):
    def __init__(self, num_classes: int = 2):
        """
        Comitê de CNNs para detecção de câncer de mama.
        num_classes=2 geralmente representa (0: Benigno/Normal, 1: Maligno)
        """
        super().__init__()

        # 1. Carregando e adaptando a ResNet-50
        self.model1: ResNet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Substitui a última camada (head) para o nosso número de classes
        num_features: int = self.model1.fc.in_features
        self.model1.fc = CustomFC(num_features, num_classes)

        # 2. Carregando e adaptando a DenseNet-121
        self.model2: DenseNet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_features: int = self.model2.classifier.in_features
        self.model2.classifier = CustomFC(num_features, num_classes)

        self.model3: VGG = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        num_features = self.model3.classifier[6].in_features
        nova_camada_final = Sequential(
            Linear(
                num_features, 256
            ),  # Additional linear layer with 256 output features
            ReLU(
                inplace=True
            ),  # Activation function (you can choose other activation functions too)
            Dropout(0.5),  # Dropout layer with 50% probability
            Linear(256, num_classes),  # Final prediction fc layer
        )
        self.model3.classifier[6] = nova_camada_final

    def init_weights(self, w_model1: str, w_model2: str, w_model3: str):
        model_data1 = load_model(self.model1, w_model1)
        # print(model_data)
        self.model1.load_state_dict(model_data1["model_weights"])

        model_data2 = load_model(self.model2, w_model2)
        # print(model_data)
        self.model2.load_state_dict(model_data2["model_weights"])

        model_data3 = load_model(self.model3, w_model3)
        # print(model_data)
        self.model3.load_state_dict(model_data3["model_weights"])

    def forward(self, x: Tensor) -> Tensor:
        # Extração dos "logits" (saídas cruas) de cada modelo
        out1: Tensor = self.model1(x)
        out2: Tensor = self.model2(x)
        out3: Tensor = self.model3(x)

        # Aplicação da função Softmax para transformar os logits em probabilidades (0 a 1)
        prob1: Tensor = F.softmax(out1, dim=1)
        prob2: Tensor = F.softmax(out2, dim=1)
        prob3: Tensor = F.softmax(out3, dim=1)

        # Soft Voting: Média das probabilidades estimadas pelos 3 modelos
        avg_prob: Tensor = (prob1 + prob2 + prob3) / 3.0

        return avg_prob
