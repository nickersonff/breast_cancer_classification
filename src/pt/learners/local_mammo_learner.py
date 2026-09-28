# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict
import logging
import os
import numpy as np
import math
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_BN_Weights
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Transposed,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    CastToTyped
)
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from src.pt.utils.dataset_torch import BreastDataset
from src.pt.preprocessing.preprocess_json import load_datalist
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from collections import Counter

class MammoLearner():
    def __init__(
        self,
        datalist: str = None,
        aggregation_epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 64,
        architecture: str = "resnet",
        conf: Dict = None,
    ):
       
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.datalist = datalist
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.best_metric = 0.0
        self.num_classes = 0
        # Epoch counter
        self.epoch_global = 0
        self.roc_values = []
        self.acc_values = []
        self.arch = architecture
        self.log = logging.getLogger(__name__)
        self.config = conf

        # The following objects will be build in `initialize()`
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.transform_valid = None
        self.sched = None

    def save_model(self, name="local_model.pt"):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights,
                     "epoch": self.epoch_global}
        model_path = os.path.join(self.config['io_dirs'].get('save_model_dir'), name)
        torch.save(save_dict, model_path) # change path

    def build_transforms(self):
        self.transform_train = T.Compose(
        [
            # Espera uma imagem já carregada em tensor no formato (C, H, W)
            T.ToDtype(torch.float32, scale=False),
            # RandRotated(range_x=pi/12, prob=0.5)
            T.RandomApply(
                [T.RandomRotation(degrees=(-15, 15))], p=0.5
            ),  # pi/12 rads = 15 graus
            # RandFlipd(spatial_axis=0, prob=0.5) -> Vertical Flip
            T.RandomVerticalFlip(p=0.5),
            # RandFlipd(spatial_axis=1, prob=0.5) -> Horizontal Flip
            T.RandomHorizontalFlip(p=0.5),
            # RandZoomd(min_zoom=0.9, max_zoom=1.1, prob=0.5)
            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=0, scale=(0.9, 1.1)
                    )  # Zoom mantendo o tamanho
                ],
                p=0.5,
            ),
            # RandGaussianSmoothd(sigma_x/y/z=(0.5, 1.15), prob=0.15)
            T.RandomApply(
                [
                    T.GaussianBlur(
                        kernel_size=(5, 5), sigma=(0.5, 1.15)
                    ) 
                ],
                p=0.15,
            ),
        ]
        )

        # 2. Transformações de Validação
        self.transform_valid = T.Compose(
            [
                T.ToDtype(torch.float32, scale=False),
            ]
        )

    def build_model(self):
        if self.arch == 'resnet':
            # RESNET18
            
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
                nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
                nn.Dropout(0.5),               # Dropout layer with 50% probability
                nn.Linear(256, self.num_classes)              # Final prediction fc layer
            )
            
        elif self.arch == 'vgg':
            # VGG16
            
            self.model = models.vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
            num_features = self.model.classifier[6].in_features
            nova_camada_final = nn.Sequential(
                nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
                nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
                nn.Dropout(0.5),               # Dropout layer with 50% probability
                nn.Linear(256, self.num_classes)              # Final prediction fc layer
            )
            self.model.classifier[6] = nova_camada_final

        elif self.arch == 'efficientnet':
            # EfficientNet B3
            
            self.model = models.efficientnet_b3(pretrained=True)
            num_features = self.model.classifier[1].in_features
            nova_camada_final = nn.Sequential(
                nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
                nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
                nn.Dropout(0.5),               # Dropout layer with 50% probability
                nn.Linear(256, self.num_classes)              # Final prediction fc layer
            )
            self.model.classifier[1] = nova_camada_final
        elif self.arch == 'resnet152':
            # RESNET152
            
            self.model = models.resnet152(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
                nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
                nn.Dropout(0.5),               # Dropout layer with 50% probability
                nn.Linear(256, self.num_classes)              # Final prediction fc layer
            )
        elif self.arch == 'densenet':
            self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            num_features = self.model.classifier.in_features

            self.model.classifier = nn.Sequential(
                    nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
                    nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
                    nn.Dropout(0.5),               # Dropout layer with 50% probability
                    nn.Linear(256, self.num_classes)              # Final prediction fc layer
            )

    def build_optimizer(self):
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),  
            eps=1e-08,            
            weight_decay=0        
        )

    def initialize(self):
        
        self.writer = SummaryWriter()

        layout = {
            "Analysis": {
                "loss": ["Multiline", ["train_loss", "val_loss"]],
                "accuracy": ["Multiline", ["train_acc", "val_acc"]],
            },
        }
        self.writer.add_custom_scalars(layout)
        self.build_transforms()

        self.num_classes = self.config['hyperparameters'].get('num_classes', 2)
        self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"Finished initializing")

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self):

        #Faz o split com o stratified group k fold 
        indices = np.arange(len(self.datalist))
        labels = np.array([item['label'] for item in self.datalist])
        # Extrai o ID do paciente de cada dicionário para formar os grupos
        groups = np.array([item['patient_id'] for item in self.datalist])

        # 3. Configura o GroupKFold (garante que o mesmo patient_id não se repita entre treino e validação)
        n_splits = 5
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


        # 4. Loop de Validação Cruzada
        # Note que passamos 'groups' no método .split()
        for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, labels, groups)):
            print(f"\n--- Início do Fold {fold + 1}/{n_splits} ---")
            print(f"Treino: {len(train_idx)} amostras, Validação: {len(val_idx)} amostras")
            # Cria as sub-listas de dicionários para este fold
            fold_train_data = [self.datalist[i] for i in train_idx]
            fold_val_data = [self.datalist[i] for i in val_idx]
            train_npy_paths = [item['npy'] for item in fold_train_data]
            train_labels = [item['label'] for item in fold_train_data]
            val_npy_paths = [item['npy'] for item in fold_val_data]
            val_labels = [item['label'] for item in fold_val_data]

            train_label_counts = Counter(train_labels)
            val_label_counts = Counter(val_labels)

            print(f"Treino - Contagem de Rótulos: {dict(train_label_counts)}")
            print(f"Validação - Contagem de Rótulos: {dict(val_label_counts)}")

            # Calcula pesos inversely proportional à frequência das classes
            total_samples = len(train_labels)
            n_classes = len(train_label_counts)

            weights = [total_samples / (n_classes * train_label_counts[c]) for c in sorted(train_label_counts.keys())]
            class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
            print(f"Pesos das classes: {class_weights}")
            # Passa o peso para a loss que você já usa no seu learner
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

            #self.criterion = torch.nn.CrossEntropyLoss()


            self.train_dataset = BreastDataset(train_npy_paths, train_labels, transform=self.transform_train)
            self.valid_dataset = BreastDataset(val_npy_paths, val_labels, transform=self.transform_valid)

            num_workers = self.config['dataloaders'].get('num_workers', 4)

            train_loader = DataLoader(self.train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=num_workers)
            valid_loader = DataLoader(self.valid_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=False, 
                                    num_workers=num_workers)
            self.build_model()
            self.model = self.model.to(self.device)
            self.build_optimizer()
            
            self.criterion = self.criterion.to(self.device)
            # Set up one-cycle learning rate scheduler
            self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr , epochs=self.aggregation_epochs,
                                                    steps_per_epoch=len(train_loader))
            
            for epoch in range(self.aggregation_epochs):            
                self.model.train()
                self.epoch_global = epoch + 1
                lrs = []
                print(
                    f"Local epoch: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
                )
                avg_loss = 0.0
                correct, total = 0,0
                for batch_idx, (images, labels) in enumerate(train_loader):
                    inputs, labels = (
                        images.to(self.device),
                        labels.to(self.device),
                    )
                    
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    #att, raw, outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    self.optimizer.step()
                    # Gradient Clipping for VGG-16
                    if self.arch == 'vgg':
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    # Record & update learning rate                
                    lrs.append(self.get_lr(self.optimizer))
                    self.sched.step()
                    avg_loss += loss.item()

                    _, _pred_label = torch.max(outputs.data, 1)
                    _labels = labels
                    total += inputs.data.size()[0]
                    correct += (_pred_label == _labels.data).sum().item()

                self.writer.add_scalar(
                    "lr", self.get_lr(self.optimizer), epoch + 1)

                self.writer.add_scalar(
                    "train_loss", avg_loss / len(train_loader), self.epoch_global)
                
                self.writer.add_scalar(
                    "train_acc", correct/float(total), self.epoch_global)

                acc, kappa, roc = self.local_valid(valid_loader)

                if len(self.acc_values) == 0:
                    self.save_model()
                elif acc >= max(self.acc_values):
                    self.save_model()
                self.roc_values.append(roc)
                self.acc_values.append(acc)
                self.writer.add_scalar("val_acc", acc, self.epoch_global)
                self.writer.add_scalar("val_kappa", kappa, self.epoch_global)

    def local_valid(
        self,
        valid_loader,
        return_probs_only=False,
        is_final=False
    ):
        if not valid_loader:
            return None
        self.model.eval()
        return_probs = []
        all_labels = []
        pred_labels = []
        l_probs = []
        val_avg_loss = 0.0
        with torch.no_grad():
            correct, total = 0, 0
            for batch_idx, (images, labels) in enumerate(valid_loader):
                inputs, lbls = (
                    images.to(self.device),
                    labels.to(self.device),
                )
                
                outputs = self.model(inputs)
                
                # Find the Loss
                validation_loss = self.criterion(outputs, lbls)
                # Calculate Loss
                val_avg_loss += validation_loss.item()
                outputs_soft = torch.softmax(outputs, dim=1)
                probs = outputs_soft.detach().cpu().numpy()
                
                # make json serializable
                for _img_file, _probs, lbl in zip(images, probs, labels):
                    p = [float(p) for p in _probs]
                    return_probs.append(
                        {
                            "image": str(_img_file),
                            "probs": p,
                            "label": int(lbl.item() if hasattr(lbl, 'item') else lbl),
                        } 
                    )
                    l_probs.append(p[1]) # probs da classe positiva
                
                if not return_probs_only:
                    _, _pred_label = torch.max(outputs_soft.data, 1)
                    _labels = labels.to(_pred_label.device)
                    total += images.data.size()[0]
                    correct += (_pred_label == _labels.data).sum().item()
                    all_labels.extend(_labels.detach().cpu().numpy())
                    pred_labels.extend(_pred_label.detach().cpu().numpy())

            self.writer.add_scalar(
                    "val_loss", (val_avg_loss/len(valid_loader)), self.epoch_global)
            
            if return_probs_only:
                return return_probs  # create a list of image names and probs
            else:
                acc = correct / float(total)
                assert len(all_labels) == total
                assert len(pred_labels) == total
                matrix = confusion_matrix(all_labels, pred_labels)
                print("### eval report ###")
                if self.num_classes == 2:
                    roc_auc = roc_auc_score(all_labels, l_probs)
                    f1 = f1_score(all_labels, pred_labels)
                    print(f'ROC Score: {roc_auc}')
                    print(f'F1-Score: {f1}')
                    
                mcc = matthews_corrcoef(all_labels, pred_labels)
                kappa = cohen_kappa_score(
                    all_labels, pred_labels, weights="linear")

                print(f'ACC: {acc}')
                print(f'MCC: {mcc}')
                print(f'Cohen Kappa Score: {kappa}')
                print(matrix)
                print('###################')

                if is_final:
                    
                    if self.num_classes == 2:
                        # ROC curve
                        fig = plt.figure(figsize=(8, 6))
                        
                        fpr, tpr, thresholds = roc_curve(all_labels, l_probs)
                        plt.plot(fpr, tpr, label='AUC = {:.4f}'.format(roc_auc))
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('ROC Curve')
                        plt.legend()
                        
                        print(f'ROC VALUES: {self.roc_values}')
                        print(f'ACC VALUES: {self.acc_values}')                        
                    
                    # CONFUSION MATRIX
                    cm_norm = []
                    cm_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=range(self.num_classes))
                    disp.plot()    

                return acc, kappa, roc_auc    