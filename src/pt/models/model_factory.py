from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torchvision.models import (
    VGG,
    DenseNet,
    DenseNet121_Weights,
    EfficientNet,
    ResNet,
    VGG16_BN_Weights,
    densenet121,
    efficientnet_b3,
    resnet18,
    resnet50,
    resnet152,
    vgg16_bn,
)

from pt.models.fccs_net import ResNet18CBAM
from pt.models.mammo_efficientnet_bilstm import MammoEfficientNetBiLSTM
from pt.models.mammo_ensemble import MammographyEnsemble
from pt.models.mv_swin_transformer import MVSwinTransformer
from pt.utils.custom_fc import CustomFC


class ModelFactory:
    """
    Fábrica responsável por centralizar a criação e customização dos modelos de Deep Learning.
    """

    @staticmethod
    def create_model(architecture: str, num_classes: int) -> Module:
        arch = architecture.lower()

        # Mapeamento de arquiteturas para métodos privados de construção
        registry = {
            "densenet": ModelFactory._build_densenet,
            "efficientnet": ModelFactory._build_efficientnet,
            "efficientnet_bilstm": ModelFactory._build_efficientnet_bilstm,
            "ensemble": ModelFactory._build_ensemble,
            "fccs-net": ModelFactory._build_fccs_net,
            "mv_swin_t": ModelFactory._build_mv_swin_t,
            "resnet": ModelFactory._build_resnet18,
            "resnet50": ModelFactory._build_resnet50,
            "resnet152": ModelFactory._build_resnet152,
            "vgg": ModelFactory._build_vgg16,
        }

        if arch not in registry:
            raise ValueError(
                f"Arquitetura '{architecture}' não suportada. Escolha entre: {list(registry.keys())}"
            )

        # Executa o método correspondente passando o número de classes
        return registry[arch](num_classes)

    @staticmethod
    def _create_custom_head(num_features: int, num_classes: int) -> Module:
        """Padroniza a cabeça de classificação para todas as redes."""
        return Sequential(
            Linear(num_features, 256),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(256, num_classes),
        )

    @staticmethod
    def _create_linear_custom_head(num_features: int, num_classes: int) -> Linear:
        """Cria uma cabeça de classificação linear simples."""
        return CustomFC(num_features, num_classes)

    @staticmethod
    def _build_resnet18(num_classes: int) -> Module:
        model: ResNet = resnet18(pretrained=True)
        num_features: int = model.fc.in_features
        model.fc = ModelFactory._create_linear_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_resnet152(num_classes: int) -> Module:
        model: ResNet = resnet152(pretrained=True)
        num_features: int = model.fc.in_features
        model.fc = ModelFactory._create_linear_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_resnet50(num_classes: int) -> Module:
        model: ResNet = resnet50(pretrained=True)
        num_features: int = model.fc.in_features
        model.fc = ModelFactory._create_linear_custom_head(num_features, num_classes)
        return model

    @staticmethod
    def _build_vgg16(num_classes: int) -> Module:
        model: VGG = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        num_features: int = model.classifier[6].in_features
        model.classifier[6] = ModelFactory._create_custom_head(
            num_features, num_classes
        )
        return model

    @staticmethod
    def _build_efficientnet(num_classes: int) -> Module:
        model: EfficientNet = efficientnet_b3(pretrained=True)
        num_features: int = model.classifier[1].in_features
        model.classifier[1] = ModelFactory._create_custom_head(
            num_features, num_classes
        )
        return model

    @staticmethod
    def _build_efficientnet_bilstm(num_classes: int) -> Module:
        return MammoEfficientNetBiLSTM(num_classes=num_classes)

    @staticmethod
    def _build_densenet(num_classes: int) -> Module:
        model: DenseNet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        num_features: int = model.classifier.in_features
        model.classifier = ModelFactory._create_linear_custom_head(
            num_features, num_classes
        )
        return model

    @staticmethod
    def _build_mv_swin_t(num_classes: int) -> Module:
        return MVSwinTransformer(num_classes=num_classes)

    @staticmethod
    def _build_fccs_net(num_classes: int) -> Module:
        return ResNet18CBAM(num_classes=num_classes)

    @staticmethod
    def _build_ensemble(num_classes: int) -> Module:
        return MammographyEnsemble(num_classes=num_classes)
