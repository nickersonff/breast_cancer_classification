import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_BN_Weights
from src.pt.models.fccs_net import ResNet18_CBAM
from src.pt.models.mammo_efficientnet_biLSTM import MammoEfficientNetBiLSTM
from src.pt.models.mammo_ensemble import MammographyEnsemble
class ModelFactory:
    """
    Fábrica responsável por centralizar a criação e customização dos modelos de Deep Learning.
    """
    @staticmethod
    def create_model(architecture: str, num_classes: int) -> nn.Module:
        arch = architecture.lower()
        
        # Mapeamento de arquiteturas para métodos 'pri'vados de construção
        registry = {
            'resnet': ModelFactory._build_resnet18,
            'resnet50': ModelFactory._build_resnet50,
            'resnet152': ModelFactory._build_resnet152,
            'vgg': ModelFactory._build_vgg16,
            'efficientnet': ModelFactory._build_efficientnet,
            'densenet': ModelFactory._build_densenet,
            'fccs-net': ModelFactory._build_fccs_net,
            'bilstm': ModelFactory._build_bilstm,
            'ensemble': ModelFactory._build_ensemble
        }

        if arch not in registry:
            raise ValueError(f"Arquitetura '{architecture}' não suportada. Escolha entre: {list(registry.keys())}")

        # Executa o método correspondente passando o número de classes
        return registry[arch](num_classes)

    @staticmethod
    def _create_custom_head(num_features: int, num_classes: int) -> nn.Module:
        """Padroniza a cabeça de classificação para todas as redes."""
        return nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    @staticmethod
    def _build_resnet18(num_classes: int) -> nn.Module:
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = ModelFactory._create_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_resnet152(num_classes: int) -> nn.Module:
        model = models.resnet152(pretrained=True)
        num_features = model.fc.in_features
        model.fc = ModelFactory._create_custom_head(num_features, num_classes)
        return model
    
    @staticmethod
    def _build_resnet50(num_classes: int) -> nn.Module:
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = ModelFactory._create_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_vgg16(num_classes: int) -> nn.Module:
        model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        num_features = model.classifier[6].in_features
        model.classifier[6] = ModelFactory._create_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_efficientnet(num_classes: int) -> nn.Module:
        model = models.efficientnet_b3(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = ModelFactory._create_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_densenet(num_classes: int) -> nn.Module:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        num_features = model.classifier.in_features
        model.classifier = ModelFactory._create_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_fccs_net(num_classes: int) -> nn.Module:
        model = ResNet18_CBAM(num_classes=num_classes)
        return model

    @staticmethod
    def _build_bilstm(num_classes: int) -> nn.Module:
        model = MammoEfficientNetBiLSTM(num_classes=num_classes, freeze_backbone=False)
        return model

    @staticmethod
    def _build_ensemble(num_classes: int) -> nn.Module:
        model = MammographyEnsemble(num_classes=num_classes)    
        return model