from torch import Tensor, flatten, softmax
from torch.nn import (
    LSTM,
    AdaptiveAvgPool2d,
    Dropout,
    Linear,
    Module,
    ReLU,
    Sequential,
    Softmax,
)
from torchvision.models import EfficientNet, EfficientNet_B0_Weights, efficientnet_b0


class TemporalAttention(Module):
    """Additive attention used to aggregate the BiLSTM sequence."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = Sequential(
            Linear(feature_dim, feature_dim),
            ReLU(inplace=True),
            Linear(feature_dim, 1),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        weights = self.score(sequence).squeeze(-1)
        weights = softmax(weights, dim=1).unsqueeze(-1)
        return (sequence * weights).sum(dim=1)


class MammoEfficientNetBiLSTM(Module):
    """
    Extrator CNN (EfficientNet-b0 pré-treinada) conectado a uma BiLSTM
    para classificação binária ou multi-classe de mamografias.

    fonte: https://www.nature.com/articles/s41598-025-95311-4.pdf

    """

    def __init__(
        self,
        num_classes: int = 2,
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if lstm_layers != 1:
            raise ValueError("A arquitetura do artigo usa exatamente uma camada BiLSTM")

        # EfficientNet-B0 pré-treinada na ImageNet.
        backbone: EfficientNet = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        # O GAP substitui o flatten espacial no bloco CNN do artigo.
        self.features: Sequential = backbone.features
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.feature_dim: int = 1280

        # A configuração publicada usa uma BiLSTM com 256 unidades.
        self.blstm = LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.attention = TemporalAttention(lstm_hidden_dim * 2)

        # A regularização L2 da camada densa é aplicada pelo weight decay.
        self.classifier = Sequential(
            Dropout(p=dropout_rate),
            Linear(lstm_hidden_dim * 2, 512),
            ReLU(inplace=True),
            Dropout(p=dropout_rate),
            Linear(512, num_classes),
        )
        self.softmax = Softmax(dim=1)

    def extract_features(self, x: Tensor) -> Tensor:
        """
        Recebe x: (B, C, H, W)
        Retorna features achatadas: (B, 1280)
        """
        feat_map: Tensor = self.features(x)
        pooled: Tensor = self.avgpool(feat_map)
        flattened: Tensor = flatten(pooled, start_dim=1)
        return flattened

    def forward(self, x: Tensor) -> Tensor:
        """
        Suporta duas dimensionalidades de entrada:
        - 5D: (Batch, Seq_Len, C, H, W) -> vistas CC/MLO do mesmo exame
        - 4D: (Batch, C, H, W) -> uma vista, com sequência de tamanho 1
        """
        if x.dim() == 5:
            batch_size, seq_len, c, h, w = x.shape
            features = self.extract_features(x.reshape(batch_size * seq_len, c, h, w))
            seq_features = features.reshape(batch_size, seq_len, self.feature_dim)
        elif x.dim() == 4:
            seq_features = self.extract_features(x).unsqueeze(1)
        else:
            raise ValueError(
                f"Dimensão de entrada esperada: 4D ou 5D. Recebido: {x.dim()}D"
            )

        lstm_out, _ = self.blstm(seq_features)
        context_vector = self.attention(lstm_out)
        return self.classifier(context_vector)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities for inference."""
        return self.softmax(self(x))
