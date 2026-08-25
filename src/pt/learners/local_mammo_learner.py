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
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_BN_Weights
from monai.data import CacheDataset, DataLoader
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
from src.pt.preprocessing.preprocess_json import load_datalist

class MammoLearner():
    def __init__(
        self,
        dataset_root: str = None,
        datalist_prefix: str = None,
        aggregation_epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 64,
        architecture: str = "resnet",
        conf: Dict = None,
    ):
       
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.dataset_root = dataset_root
        self.datalist_prefix = datalist_prefix
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.best_metric = 0.0
        self.run = None
        self.num_classes = 0
        # Epoch counter
        self.epoch_global = 0
        self.roc_values = []
        self.acc_values = []
        self.arch = architecture

        # The following objects will be build in `initialize()`
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.train_loader = None
        self.valid_dataset = None
        self.valid_loader = None
        self.sched = None
        self.config = None
        self.log = logging.getLogger(__name__)
        self.config = conf

    def save_model(self, name="local_model.pt"):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights,
                     "epoch": self.epoch_global}
        model_path = os.path.join(self.config['io_dirs'].get('save_model_dir'), name)
        torch.save(save_dict, model_path) # change path

    def build_transforms(self):
        self.transform_train = Compose(
            [
                LoadImaged(keys=["image"]),
                RandRotated(keys=["image"], range_x=np.pi / \
                            12, prob=0.5, keep_size=True),
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
                RandZoomd(keys=["image"], min_zoom=0.9,
                        max_zoom=1.1, prob=0.5, keep_size=True),
                RandGaussianSmoothd(
                    keys=["image"],
                    sigma_x=(0.5, 1.15),
                    sigma_y=(0.5, 1.15),
                    sigma_z=(0.5, 1.15),
                    prob=0.15,
                ),
                RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
                RandGaussianNoised(keys=["image"], std=0.01, prob=0.15),
                # make channels-first
                Transposed(keys=["image"], indices=[2, 0, 1]),
                CastToTyped(keys=["image"], dtype=torch.float32),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        
        self.transform_valid = Compose(
            [
                LoadImaged(keys=["image"]),
                # make channels-first
                Transposed(keys=["image"], indices=[2, 0, 1]),
                CastToTyped(keys=["image"], dtype=torch.float32),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def build_dataloaders(self):
        # Note, do not change this syntax. The data list filename is given by the system.
        datalist_file = self.datalist_prefix 
        if not os.path.isfile(datalist_file):
            print(f"{datalist_file} does not exist!")

        # Set dataset
        train_datalist = load_datalist(
            datalist_file,
            data_list_key="train",  # do not change this key name
            base_dir=self.dataset_root,
        )
        
        val_datalist = load_datalist(
            datalist_file,
            data_list_key="test",
            base_dir=self.dataset_root,
        )

        num_workers = self.config['dataloaders'].get('num_workers', 4)  
        cache_rate = self.config['dataloaders'].get('cache_rate', 1.0)  

        self.train_dataset = CacheDataset(
            data=train_datalist,
            transform=self.transform_train,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        print( f"Training set: {len(train_datalist)} entries")

        if len(val_datalist) > 0:
            self.valid_dataset = CacheDataset(
                data=val_datalist,
                transform=self.transform_valid,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
            self.valid_loader = DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            print( 
                f"Validation set: {len(val_datalist)} entries")
        else:
            self.valid_dataset = None
            self.valid_loader = None
            print("Use no validation set")

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
        
        self.build_dataloaders()
        
        self.build_model()

        self.model = self.model.to(self.device)
        if self.optimizer == None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.criterion = self.criterion.to(self.device)

        # Set up one-cycle learning rate scheduler
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr , epochs=self.aggregation_epochs,
                                                steps_per_epoch=len(self.train_loader))

        print(f"Finished initializing")

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, train_loader):
        
        for epoch in range(self.aggregation_epochs):            
            self.model.train()
            self.epoch_global = epoch + 1
            lrs = []
            print(
                f"Local epoch: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )
            avg_loss = 0.0
            correct, total = 0,0
            for i, batch_data in enumerate(train_loader):
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )
                
                # Gradient Clipping for VGG-16
                if self.arch == 'vgg':
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                #att, raw, outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()

                # Record & update learning rate                
                lrs.append(self.get_lr(self.optimizer))
                self.sched.step()
                avg_loss += loss.item()

                
                _, _pred_label = torch.max(outputs.data, 1)
                _labels = batch_data["label"].to(self.device)
                total += inputs.data.size()[0]
                correct += (_pred_label == _labels.data).sum().item()

            self.writer.add_scalar(
                "lr", self.get_lr(self.optimizer), epoch + 1)

            self.writer.add_scalar(
                "train_loss", avg_loss / len(train_loader), self.epoch_global)
            
            self.writer.add_scalar(
                "train_acc", correct/float(total), self.epoch_global)

            acc, kappa, roc = self.local_valid(
                self.valid_loader
            )

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
        labels = []
        pred_labels = []
        l_probs = []
        val_avg_loss = 0.0
        with torch.no_grad():
            correct, total = 0, 0
            for i, batch_data in enumerate(valid_loader):
                inputs, lbls = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )
                
                outputs = self.model(inputs)
                
                # Find the Loss
                validation_loss = self.criterion(outputs, lbls)
                # Calculate Loss
                val_avg_loss += validation_loss.item()
                outputs_soft = torch.softmax(outputs, dim=1)
                probs = outputs_soft.detach().cpu().numpy()
                
                # make json serializable
                for _img_file, _probs, lbl in zip(batch_data["image"].meta["filename_or_obj"], probs, batch_data["label"]):
                    p = [float(p) for p in _probs]
                    return_probs.append(
                        {
                            "image": os.path.basename(_img_file),
                            "probs": p,
                            "label": lbl,
                        } 
                    )
                    l_probs.append(p[1]) # probs da classe positiva
                
                if not return_probs_only:
                    _, _pred_label = torch.max(outputs_soft.data, 1)
                    _labels = batch_data["label"].to(self.device)
                    total += inputs.data.size()[0]
                    correct += (_pred_label == _labels.data).sum().item()
                    labels.extend(_labels.detach().cpu().numpy())
                    pred_labels.extend(_pred_label.detach().cpu().numpy())

            self.writer.add_scalar(
                    "val_loss", (val_avg_loss/len(valid_loader)), self.epoch_global)
            
            if return_probs_only:
                return return_probs  # create a list of image names and probs
            else:
                acc = correct / float(total)
                assert len(labels) == total
                assert len(pred_labels) == total
                matrix = confusion_matrix(labels, pred_labels)
                print("### eval report ###")
                if self.num_classes == 2:
                    roc_auc = roc_auc_score(labels, l_probs)
                    f1 = f1_score(labels, pred_labels)
                    print(f'ROC Score: {roc_auc}')
                    print(f'F1-Score: {f1}')
                    
                mcc = matthews_corrcoef(labels, pred_labels)
                kappa = cohen_kappa_score(
                    labels, pred_labels, weights="linear")

                print(f'ACC: {acc}')
                print(f'MCC: {mcc}')
                print(f'Cohen Kappa Score: {kappa}')
                print(matrix)
                print('###################')

                if is_final:
                    
                    if self.num_classes == 2:
                        # ROC curve
                        fig = plt.figure(figsize=(8, 6))
                        
                        fpr, tpr, thresholds = roc_curve(labels, l_probs)
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