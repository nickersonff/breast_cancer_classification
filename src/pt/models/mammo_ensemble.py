import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torchvision.models import VGG16_BN_Weights


class MammographyEnsemble(nn.Module):
    def __init__(self, num_classes=2):
        """
        Comitê de CNNs para detecção de câncer de mama.
        num_classes=2 geralmente representa (0: Benigno/Normal, 1: Maligno)
        """
        super(MammographyEnsemble, self).__init__()

        # 1. Carregando e adaptando a ResNet-50
        self.model1 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Substitui a última camada (head) para o nosso número de classes
        num_features = self.model1.fc.in_features
        self.model1.fc = nn.Sequential(
            nn.Linear(
                num_features, 256
            ),  # Additional linear layer with 256 output features
            nn.ReLU(
                inplace=True
            ),  # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),  # Dropout layer with 50% probability
            nn.Linear(256, num_classes),  # Final prediction fc layer
        )

        # 2. Carregando e adaptando a DenseNet-121
        self.model2 = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = self.model2.classifier.in_features
        self.model2.classifier = nn.Sequential(
            nn.Linear(
                num_features, 256
            ),  # Additional linear layer with 256 output features
            nn.ReLU(
                inplace=True
            ),  # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),  # Dropout layer with 50% probability
            nn.Linear(256, num_classes),  # Final prediction fc layer
        )

        self.model3 = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        num_features = self.model3.classifier[6].in_features
        nova_camada_final = nn.Sequential(
            nn.Linear(
                num_features, 256
            ),  # Additional linear layer with 256 output features
            nn.ReLU(
                inplace=True
            ),  # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),  # Dropout layer with 50% probability
            nn.Linear(256, num_classes),  # Final prediction fc layer
        )
        self.model3.classifier[6] = nova_camada_final

    def init_weights(self, w_model1, w_model2, w_model3):
        model_data1 = torch.load(w_model1)
        # print(model_data)
        self.model1.load_state_dict(model_data1["model_weights"])

        model_data2 = torch.load(w_model2)
        # print(model_data)
        self.model2.load_state_dict(model_data2["model_weights"])

        model_data3 = torch.load(w_model3)
        # print(model_data)
        self.model3.load_state_dict(model_data3["model_weights"])

    def forward(self, x):
        # Extração dos "logits" (saídas cruas) de cada modelo
        out1 = self.model1(x)
        out2 = self.model2(x)
        out3 = self.model3(x)

        # Aplicação da função Softmax para transformar os logits em probabilidades (0 a 1)
        prob1 = F.softmax(out1, dim=1)
        prob2 = F.softmax(out2, dim=1)
        prob3 = F.softmax(out3, dim=1)

        # Soft Voting: Média das probabilidades estimadas pelos 3 modelos
        avg_prob = (prob1 + prob2 + prob3) / 3.0

        return avg_prob
