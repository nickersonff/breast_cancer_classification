import os

import numpy as np
import torch
from torch.utils.data import Dataset

from pt.utils.constants import Constants


class BreastDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # ⚠️ ADICIONE ESTE MÉTODO: Retorna o número total de itens/amostras do dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Carrega a imagem do disco
        # Se a imagem original for NumPy array em formato (H, W, C):
        config = Constants.get_config()
        project_root = Constants.get_absolute_project_path()
        full_path = os.path.join(
            project_root,
            config["io_dirs"].get("preprocess_prefix"),
            self.image_paths[idx],
        )
        img_array = np.load(full_path)
        img_tensor = torch.from_numpy(img_array).permute(
            2, 0, 1
        )  # Equivalente ao Transposed([2, 0, 1])

        label = torch.tensor(self.labels[idx])

        # 2. Aplica o Compose
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label
