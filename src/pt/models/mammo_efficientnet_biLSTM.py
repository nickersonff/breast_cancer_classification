import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class MammoEfficientNetBiLSTM(nn.Module):
    """
    Extrator CNN (EfficientNet-b0 pré-treinada) conectado a uma BiLSTM
    para classificação binária ou multi-classe de mamografias.

    fonte: https://www.nature.com/articles/s41598-025-95311-4.pdf

    """
    def __init__(
        self,
        num_classes: int = 1,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        dropout_rate: float = 0.3,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        # 1. Carrega EfficientNet-B0 pré-treinada na ImageNet
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = efficientnet_b0(weights=weights)
        
        # Opcional: congelar pesos para Feature Extraction pura
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
                
        # Mantém apenas as camadas convolucionais (features) e o pooling global adaptativo
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # EfficientNet-B0 gera 1280 canais no topo
        self.feature_dim = 1280
        
        # 2. Camada Bidirectional LSTM
        # Entrada: (batch_size, seq_len, feature_dim)
        self.blstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )
        
        # 3. Cabeça de Classificação
        # Como é bidirecional, a dimensão de saída é 2 * lstm_hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(lstm_hidden_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Recebe x: (B, C, H, W)
        Retorna features achatadas: (B, 1280)
        """
        feat_map = self.features(x)
        pooled = self.avgpool(feat_map)
        flattened = torch.flatten(pooled, start_dim=1)
        return flattened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Suporta duas dimensionalidades de entrada:
        - 5D: (Batch, Seq_Len, C, H, W) -> caso multi-view (ex.: CC + MLO)
        - 4D: (Batch, C, H, W) -> divide mapa espacial em sequência
        """
        if x.dim() == 5:
            # Caso Multi-View (ex: 2 ou 4 projeções mamográficas por exame)
            batch_size, seq_len, c, h, w = x.shape
            
            # Achata batch e sequência para passar pela CNN em paralelo
            x_reshaped = x.view(batch_size * seq_len, c, h, w)
            features = self.extract_features(x_reshaped)  # Shape: (B * S, 1280)
            
            # Reconstrói a sequência para a BiLSTM: (Batch, Seq_Len, 1280)
            seq_features = features.view(batch_size, seq_len, -1)
            
        elif x.dim() == 4:
            # Caso Imagem Única: trata canais espaciais como sequência
            # Passa pelas camadas convolucionais: (Batch, 1280, H', W')
            feat_map = self.features(x)
            batch_size, channels, h, w = feat_map.shape
            
            # Achata as dimensões espaciais em uma sequência de tamanho H'*W'
            # (Batch, channels, H'*W') -> permuta para (Batch, H'*W', channels)
            seq_features = feat_map.view(batch_size, channels, -1).permute(0, 2, 1)
        else:
            raise ValueError(f"Dimensão de entrada esperada: 4D ou 5D. Recebido: {x.dim()}D")

        # Passa pela BiLSTM -> out shape: (Batch, Seq_Len, hidden_dim * 2)
        lstm_out, (h_n, c_n) = self.blstm(seq_features)
        
        # Agregação temporal via Global Average Pooling sobre os passos
        # Alternativamente, pode-se concatenar h_n[-2,:,:] e h_n[-1,:,:]
        context_vector = torch.mean(lstm_out, dim=1)  # Shape: (Batch, hidden_dim * 2)
        
        # Predição final
        logits = self.classifier(context_vector)
        return logits