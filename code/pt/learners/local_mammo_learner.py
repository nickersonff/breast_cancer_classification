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
import json
import sys
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_BN_Weights
import monai.utils as mutil
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import TorchVisionFCModel
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
    HistogramNormalized,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    CastToTyped
)
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from torch.utils.tensorboard import SummaryWriter
import wandb
from types import SimpleNamespace
import matplotlib.pyplot as plt
from utils.preprocess_json import preprocessMixedDB
import random

def load_datalist(filename, data_list_key="train", base_dir=""):
    with open(filename, "r") as f:
        data = json.load(f)

    data_list = data[data_list_key]
    for d in data_list:
        d["image"] = os.path.join(base_dir, d["image"])

    return data_list

class MammoLearner():
    def __init__(
        self,
        dataset_root: str = None,
        datalist_prefix: str = None,
        aggregation_epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 64,
        val_freq: int = 1,
        val_frac: float = 0.1,
        architecture: str = "resnet",
    ):
       
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.dataset_root = dataset_root
        self.datalist_prefix = datalist_prefix
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.best_metric = 0.0
        self.val_frac = val_frac
        self.run = None
        self.num_classes = 0
        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0
        self.roc_values = []
        self.acc_values = []
        self.arch = architecture


        if not isinstance(self.val_freq, int):
            raise ValueError(
                f"Expected `val_freq` but got type {type(self.val_freq)}")

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
        self.run = None
        
    # to use wandb
    def init_wandb(self):

        wandb.login()
        config = SimpleNamespace(
            batch_size=self.batch_size,
            learning_rate=self.lr,
            epochs=self.aggregation_epochs,
            num_workers=4,
            model_name='resnest18',
            num_classes=2,
            in_chans=1,
            device = "cuda:0" if torch.cuda.is_available() else "cpu",
            link_model=True            
        )

        self.run = wandb.init(project="wandb_mammo",
                     job_type="train", 
                     sync_tensorboard=True,
                     config=config 
                     )

    def save_model(self, name="local_model.pt"):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights,
                     "epoch": self.epoch_global}
       
        torch.save(save_dict, f'/home/nfferreira/data/model/{name}') # change path

    def initialize(self):
        
        self.writer = SummaryWriter()

        layout = {
            "Analysis": {
                "loss": ["Multiline", ["train_loss", "val_loss"]],
                "accuracy": ["Multiline", ["train_acc", "val_acc"]],
            },
        }
        self.writer.add_custom_scalars(layout)
        
        
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
       
        y = torch.tensor([d['label'] for d in train_datalist])
        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        
        val_datalist = load_datalist(
            datalist_file,
            data_list_key="test",
            base_dir=self.dataset_root,
        )

        num_workers = 4  # tuned for challenge system. Please do not change.
        cache_rate = 1.0
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
        
        self.num_classes = len(np.unique(y))

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

        self.model = self.model.to(self.device)
        if self.optimizer == None:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.criterion = self.criterion.to(self.device)

        # Set up one-cycle learning rate scheduler
        self.sched = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr , epochs=self.aggregation_epochs,
                                                steps_per_epoch=len(self.train_loader))

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

        print(f"Finished initializing")

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, train_loader):
        
        for epoch in range(self.aggregation_epochs):            
            self.model.train()
            self.epoch_global = self.epoch_of_start_time + epoch
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
                "lr", self.get_lr(self.optimizer), self.epoch_global)

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

            if self.run != None:
                wandb.log(
                    {
                        "epoch": self.epoch_global,
                        "train_acc": (correct/float(total)),
                        "train_loss": (avg_loss / len(train_loader)),
                        "val_acc": acc,
                        "val_kappa": kappa
                    }
                )
            
    

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
                current_step = len(valid_loader) * self.epoch_global + i
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
                        
                        if self.run != None:
                            self.run.log(
                                {
                                    "ROC CURVE IMAGE": wandb.Image(fig, caption='ROC CURVE')
                                }
                            )
                        print(f'ROC VALUES: {self.roc_values}')
                        print(f'ACC VALUES: {self.acc_values}')
                        
                    
                    # CONFUSION MATRIX
                    cm_norm = []
                    cm_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=range(self.num_classes))
                    disp.plot()    
                    
                    if self.run != None: 
                        self.run.log(
                                {
                                    "CONFUSION MATRIX IMAGE": wandb.Image(disp.figure_, caption='CONFUSION MATRIX')
                                }
                            )

                return acc, kappa, roc_auc    


def runTest(dataset_root, datalist_prefix, batch=64, cnn='resnet', fine=False):
    print("Testing MammoLearner...")
    learner = MammoLearner(
        dataset_root=dataset_root,
        datalist_prefix=datalist_prefix,
        aggregation_epochs=60,
        val_frac=0,
        lr=1e-3,
        batch_size=batch,
        architecture=cnn,
    )
    print("test initialize...")
    learner.initialize()

    print("test train...")
    learner.train(
        train_loader=learner.train_loader
    )
    
    learner.save_model('final-model.pt')

    print("test valid...")
    acc, kappa, roc = learner.local_valid(
        valid_loader=learner.valid_loader, is_final=True
    )
    
    print("debug acc", acc)
    print("debug kappa", kappa)
    print("debug ROC AUC", roc)
        
    if learner.run != None:
        learner.run.finish()

def preprocessing(debug_datalist='/home/nfferreira/data/dataset_site-1.json', 
                      debug_dataset_root = "/home/nfferreira/data/preprocessed",
                      cnn='resnet', fineT=False):
    
    print(f'ARQUIVO: {debug_datalist}')
    """
    DEFAULT PIPELINE - NO NORMALIZATION - NO FILTERS - 224 X 224
    """
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-NA_FILTER-NA_SIZE-224/'
    print(f'**** Pipeline: DEFAULT PIPELINE - NO NORMALIZATION - NO FILTERS - 224 X 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=16, cnn=cnn, fine=fineT)

    """
    MIN-MAX NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-NA_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-NA_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024, norm="min-max", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    Z-SCORE NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-z-score_FILTER-NA_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-z-score_FILTER-NA_SIZE-1024/'
    print(f'**** Pipeline: Z-SCORE NORMALIZATION PIPELINE - NO FILTERS - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024, norm="z-score", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024,norm="min-max", filter="CLAHE", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - GAUSSIAN - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-GAUSSIAN_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-GAUSSIAN_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - GAUSSIAN - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root,size=1024, norm="min-max", filter="GAUSSIAN", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - BILATERAL - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-BILATERAL_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-BILATERAL_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - BILATERAL - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root,size=1024, norm="min-max", filter="BILATERAL", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - WIENER - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-WIENER_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-WIENER_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - WIENER - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024,norm="min-max", filter="WIENER", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - MEDIAN - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-MEDIAN_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-MEDIAN_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - MEDIAN - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024,norm="min-max", filter="MEDIAN", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+BILATERAL - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+BILATERAL_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+BILATERAL_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+BILATERAL - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024,norm="min-max", filter="CLAHE+BILATERAL", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+GAUSSIAN - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+GAUSSIAN_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+GAUSSIAN_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+GAUSSIAN - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root,size=1024, norm="min-max", filter="CLAHE+GAUSSIAN", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+WIENER - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+WIENER_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+WIENER_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+WIENER - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root,size=1024, norm="min-max",filter="CLAHE+WIENER", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    MIN-MAX NORMALIZATION PIPELINE - CLAHE+MEDIAN - 1024 X 1024
    """
    #debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+MEDIAN_SIZE-224/'
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-min-max_FILTER-CLAHE+MEDIAN_SIZE-1024/'
    print(f'**** Pipeline: MIN-MAX NORMALIZATION PIPELINE - CLAHE+MEDIAN - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root,size=1024, norm="min-max", filter="CLAHE+MEDIAN", datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 384 X 384
    """
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-NA_FILTER-NA_SIZE-384/'
    print(f'**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 384 X 384 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=384, datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=32, cnn=cnn, fine=fineT)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 512 X 512
    """
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-NA_FILTER-NA_SIZE-512/'
    print(f'**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 512 X 512 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=512, datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 1024 X 1024
    """
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-NA_FILTER-NA_SIZE-1024/'
    print(f'**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 1024 X 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=1024, datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=8, cnn=cnn, fine=fineT)
    """
    RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 2048 X 2048
    """
    debug_dataset_root = f'/home/nfferreira/data/preprocessed/DDSM2_NORM-NA_FILTER-NA_SIZE-2048/'
    print(f'**** Pipeline: RESIZE PIPELINE - NO NORMALIZATION - NO FILTER - 2048 X 2048 ****')
    preprocessMixedDB(out_path=debug_dataset_root, size=2048, datalist=debug_datalist)
    runTest(debug_dataset_root, debug_datalist, batch=2, cnn=cnn, fine=fineT)


def path_exists(caminho_da_pasta=""):

    if os.path.exists(caminho_da_pasta) and os.path.isdir(caminho_da_pasta):
        if os.listdir(caminho_da_pasta):
            return True
        else:
            return False
    else:
        return False


def pipelines(debug_datalist = "/home/nfferreira/data/dataset_site-1.json", 
              cnn='resnet'):

    dic = {"/home/nfferreira/data/dataset_site-1.json": "DDSM",
           "/home/nfferreira/data/dataset_site-1-Planmed.json": "PLANMED",
           "/home/nfferreira/data/dataset_site-1-IMS.json": "IMS",
           "/home/nfferreira/data/dataset_site-1-SIEMENS.json": "SIEMENS",
           "/home/nfferreira/data/dataset_site-1_VINDR_DDSM-reduzido.json": "VINDR-DDSM",
           "/home/nfferreira/data/dataset_site-1-VINDR_ALLMAN.json": "VINDR"}

    normalizacao = ['min-max', 'z-score']
    filtros = ['CLAHE', 'BILATERAL', 'WIENER', 'GAUSSIAN', 
               'MEDIAN', 'CLAHE+BILATERAL', 
    'CLAHE+WIENER', 'CLAHE+GAUSSIAN', 'CLAHE+MEDIAN']
    tamanhos = [224, 384, 512, 1024, 2048]
    pipe = []

    random.seed(42)
    qt_exec = 25
    
    while len(pipe) < qt_exec:
        t = (random.sample(range(len(normalizacao)), k=1)[0],
            random.sample(range(len(filtros)), k=1)[0],
            random.sample(range(len(tamanhos)), k=1)[0] )
        if (t not in pipe):
            pipe.append(t)

    for i in pipe:
        outpath = f'/home/nfferreira/data2/preprocessed/{dic[debug_datalist]}_NORM-{normalizacao[i[0]]}_FILTER-{filtros[i[1]]}_SIZE-{tamanhos[i[2]]}/'
        print(outpath)
        if not path_exists(outpath):
            print(f'**** Pipeline: {normalizacao[i[0]]} - {filtros[i[1]]} - {tamanhos[i[2]]} ****')
            #os.mkdir(outpath)
            preprocessMixedDB(out_path=outpath, norm=normalizacao[i[0]], filter=filtros[i[1]], 
                    size=tamanhos[i[2]], datalist=debug_datalist)
        
        
        if tamanhos[i[2]] == 2048:
            runTest(outpath, debug_datalist, batch=4, cnn=cnn)
        elif tamanhos[i[2]] == 1024:
            runTest(outpath, debug_datalist, batch=16, cnn=cnn)
        else: 
            runTest(outpath, debug_datalist, batch=32, cnn=cnn)


if __name__ == "__main__":
    
    argumentos = sys.argv

    # Modify the datalist paths according to your local environment
    datalist = ["/home/nfferreira/data/dataset_site-1.json",
           "/home/nfferreira/data/dataset_site-1-Planmed.json",
           "/home/nfferreira/data/dataset_site-1-IMS.json",
           "/home/nfferreira/data/dataset_site-1-SIEMENS.json",
           "/home/nfferreira/data/dataset_site-1_VINDR_DDSM-reduzido.json",
           "/home/nfferreira/data/dataset_site-1-VINDR_ALLMAN.json"
           ]
    
    cnn = argumentos[2]

    if argumentos[1]=='preprocess':
        for i in range(0,10):
            dataset = f'/home/nfferreira/data/dataset_site-{i}_DDSM_KFOLD.json'
            preprocessing(debug_datalist=dataset, cnn=cnn)

    elif argumentos[1]=='pipelines':
        for i in datalist:
            pipelines(debug_datalist=i, cnn=cnn)
    