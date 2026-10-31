from torch import Tensor
from torch.nn import Dropout, Linear, ReLU, Sequential


class CustomFC(Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features)

        self.block = Sequential(
            Linear(in_features, 256),
            ReLU(inplace=True),
            Dropout(0.5),
            Linear(256, out_features),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)
