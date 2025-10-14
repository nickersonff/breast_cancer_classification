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
)
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from torch.utils.tensorboard import SummaryWriter
import wandb
from types import SimpleNamespace
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from utils.preprocess_json import preprocess, preprocessMixedDB
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
        
    def save_model(self, name="local_model.pt"):
        # save model
        model_weights = self.model.state_dict()
        save_dict = {"model_weights": model_weights,
                     "epoch": self.epoch_global}
       
        torch.save(save_dict, f'/home/nfferreira/data/model/{name}') # change the path

    def initialize(self, fine_tunning=False):
        
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
                HistogramNormalized(keys=["image"]),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        
        self.transform_valid = Compose(
            [
                LoadImaged(keys=["image"]),
                # make channels-first
                Transposed(keys=["image"], indices=[2, 0, 1]),
                HistogramNormalized(keys=["image"]),
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
        self.model = models.resnet18(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
            nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
            nn.Dropout(0.5),               # Dropout layer with 50% probability
            nn.Linear(256, 2)              # Final prediction fc layer
        )
        num_classes = 2  
        
        if fine_tunning:
            #################### BLOCO PARA O FINE-TUNING
            # carregar os pesos de um modelo treinado
            model_data = torch.load("/home/nfferreira/data/model/final_model.pt") # change path
            self.model.load_state_dict(model_data['model_weights'])

            # To freeze the residual layers
            for param in self.model.parameters():
                param.requires_grad = False
                
            camadas = [self.model.layer4, self.model.fc]
            for c in camadas:
                for param in c.parameters():
                    param.requires_grad = True
            

        self.model = self.model.to(self.device)
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

                # Escrever os pesos do modelo no TensorBoard
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

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

            
    def fine_tunning(self, train_loader):
        
        # treina a base do modelo com as camadas congeladas por 20 epochs
        for epoch in range(self.aggregation_epochs):
            print(f'Treino na camada Mid-Level: {epoch+1}/{self.aggregation_epochs}')
            self.model.train()
            avg_loss = 0.0
            correct, total = 0,0
            for i, batch_data in enumerate(train_loader):
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                self.sched.step()
                avg_loss += loss.item()

                _, _pred_label = torch.max(outputs.data, 1)
                _labels = batch_data["label"].to(self.device)
                total += inputs.data.size()[0]
                correct += (_pred_label == _labels.data).sum().item()

                # Escrever os pesos do modelo no TensorBoard
                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

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
                #att, raw, outputs = self.model(inputs)
                
                # Find the Loss
                validation_loss = self.criterion(outputs, lbls)
                # Calculate Loss
                current_step = len(valid_loader) * self.epoch_global + i
                val_avg_loss += validation_loss.item()
                outputs_soft = torch.softmax(outputs, dim=1)
                probs = outputs_soft.detach().cpu().numpy()
                
                # make json serializable
                for _img_file, _probs, lbl in zip(batch_data["image_meta_dict"]["filename_or_obj"], probs, batch_data["label"]):
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
                        # gera a curva roc
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
                        
                    
                    # gera a matriz de confusão
                    cm_norm = []
                    cm_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=range(self.num_classes))
                    disp.plot()    

                return acc, kappa, roc_auc    

def run_test_case(dataset_root, datalist_prefix, batch=64):
    print("Testing MammoLearner...")
    learner = MammoLearner(
        dataset_root=dataset_root,
        datalist_prefix=datalist_prefix,
        aggregation_epochs=60,
        val_frac=0,
        lr=1e-3,
        batch_size=batch
    )
    print("test initialize...")
    fine = False
    learner.initialize(fine_tunning=fine)
    if fine:
        print("test fine-tunning...")
        learner.fine_tunning(
            train_loader=learner.train_loader
        )
    else:
        print("test train...")
        learner.train(
            train_loader=learner.train_loader
        )

    learner.save_model('final_model.pt')

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
                      debug_dataset_root = "/home/nfferreira/data/preprocessed"):
    #NPAD
    dicom_root = "/home/nfferreira/DDSM/CBIS-DDSM"
    print(f'Dataset file: {debug_datalist}')
    """
    CENÁRIO DE TESTES PARA OS PARÂMETROS FIXOS E SEM NENHUM OUTRO PRE-PROCESSAMENTO
    """
    print(f'**** Pipeline: SEM NORMALIZAÇÃO - SEM FILTROS - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist)
    run_test_case(debug_dataset_root, debug_datalist, batch=16)

    """
    CENÁRIO DE TESTES PARA NORMALIZAÇÃO MIN-MAX
    """
    print(f'**** Pipeline: MIN-MAX - SEM FILTROS - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA NORMALIZAÇÃO Z-SCORE
    """
    print(f'**** Pipeline: Z-SCORE - SEM FILTROS - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="z-score")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - CLAHE
    """
    print(f'**** Pipeline: MIN-MAX - CLAHE - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max", filter="CLAHE")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - GAUSSIAN
    """
    print(f'**** Pipeline: MIN-MAX - GAUSSIAN - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="GAUSSIAN")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - BILATERAL
    """
    print(f'**** Pipeline: MIN-MAX - BILATERAL - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="BILATERAL")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - WIENER
    """
    print(f'**** Pipeline: MIN-MAX - WIENER - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="WIENER")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - MEDIAN
    """
    print(f'**** Pipeline: MIN-MAX - MEDIAN - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="MEDIAN")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - CLAHE+BILATERAL
    """
    print(f'**** Pipeline: MIN-MAX - CLAHE+BILATERAL - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="CLAHE+BILATERAL")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - CLAHE+GAUSSIAN
    """
    print(f'**** Pipeline: MIN-MAX - CLAHE+GAUSSIAN - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="CLAHE+GAUSSIAN")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA FILTROS - CLAHE+WIENER
    """
    print(f'**** Pipeline: MIN-MAX - CLAHE+WIENER - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, norm="min-max",filter="CLAHE+WIENER", 
                     datalist=debug_datalist)
    run_test_case(debug_dataset_root, debug_datalist, batch=16) 
    """
    CENÁRIO DE TESTES PARA FILTROS - CLAHE+MEDIAN
    """
    print(f'**** Pipeline: MIN-MAX - CLAHE+MEDIAN - 224 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, norm="min-max",filter="CLAHE+MEDIAN")
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA TAMANHOS - 384x384
    """
    print(f'**** Pipeline: SEM NORMALIZAÇÃO - SEM FILTRO - 384 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, size=384)
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA TAMANHOS - 512x512
    """
    print(f'**** Pipeline: SEM NORMALIZAÇÃO - SEM FILTRO - 512 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, size=512)
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA TAMANHOS - 1024x1024
    """
    print(f'**** Pipeline: SEM NORMALIZAÇÃO - SEM FILTRO - 1024 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, size=1024)
    run_test_case(debug_dataset_root, debug_datalist, batch=16)
    """
    CENÁRIO DE TESTES PARA TAMANHOS - 2048x2048
    """
    print(f'**** Pipeline: SEM NORMALIZAÇÃO - SEM FILTRO - 2048 ****')
    preprocessMixedDB(out_path=debug_dataset_root, datalist=debug_datalist, size=2048)
    run_test_case(debug_dataset_root, debug_datalist, batch=2)
    """
    CENÁRIO DE TESTES COM OS MELHORES RESULTADOS DOS TESTES ANTERIORES (OPP)
    """
    #print(f'**** Pipeline BEST: MIN-MAX - CLAHE+GAUSSIAN - 2048 ****')
    #preprocessMixedDB(out_path=debug_dataset_root, norm="min-max",filter="CLAHE+GAUSSIAN", size=2048, 
    #                  datalist=debug_datalist)
    #run_test_case(debug_dataset_root, debug_datalist, batch=2)

def pipelines(debug_datalist = "/home/nfferreira/data/dataset_site-1.json", 
              debug_dataset_root = "/home/nfferreira/data/preprocessed-2"):

    normalizacao = ['min-max', 'z-score']
    filtros = ['CLAHE', 'BILATERAL', 'WIENER', 'GAUSSIAN', 
               'MEDIAN', 'CLAHE+BILATERAL', 
    'CLAHE+WIENER', 'CLAHE+GAUSSIAN', 'CLAHE+MEDIAN']
    tamanhos = [224, 384, 512, 1024, 2048]
    pipe = []
    exc = []  # se já rodou essas configs

    random.seed(42)

    while len(pipe) < 25:
        t = (random.sample(range(len(normalizacao)), k=1)[0],
            random.sample(range(len(filtros)), k=1)[0],
            random.sample(range(len(tamanhos)), k=1)[0] )
        if (t not in pipe) and (t not in exc):
            pipe.append(t)

    for i in pipe:
        print(f'**** Pipeline: {normalizacao[i[0]]} - {filtros[i[1]]} - {tamanhos[i[2]]} ****')
        preprocessMixedDB(out_path=debug_dataset_root, norm=normalizacao[i[0]], filter=filtros[i[1]], 
                   size=tamanhos[i[2]], datalist=debug_datalist)
        if tamanhos[i[2]] == 2048:
            run_test_case(debug_dataset_root, debug_datalist, batch=2)
        else:
            run_test_case(debug_dataset_root, debug_datalist, batch=16)

if __name__ == "__main__":
    #EXECUTA TODOS OS EXPERIMENTOS SOBRE OS PRE-PROCESSAMENTOS - NORMALIZAÇÃO, FILTROS E TAMANHOS  
    argumentos = sys.argv
    #pipelines(debug_datalist=argumentos[1], debug_dataset_root=argumentos[2])
    preprocessing(debug_datalist=argumentos[1], debug_dataset_root=argumentos[2])
    
    #lista = ['/home/nfferreira/data/dataset_site-1_VINDR_DDSM-reduzido.json']
    #for i in lista:
        #dataset = f'/home/nfferreira/data/dataset_site-{i}_DDSM_KFOLD.json'
    #    preprocessing(debug_datalist=i, debug_dataset_root=argumentos[2])
    
