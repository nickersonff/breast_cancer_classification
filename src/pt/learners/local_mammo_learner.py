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
from logging import Logger, getLogger
from os import makedirs
from os.path import basename, isfile, join
from typing import Any

import numpy as np
from matplotlib.pyplot import legend, plot, show, title, xlabel, xlim, ylabel, ylim
from monai.data.dataloader import DataLoader
from monai.data.dataset import CacheDataset
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.transforms.spatial.dictionary import RandFlipd, RandRotated, RandZoomd
from monai.transforms.utility.dictionary import CastToTyped, EnsureTyped, Transposed
from numpy import pi
from safetensors.torch import save_model
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from torch import Tensor, device, float32, max as max_torch, no_grad, softmax
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss, Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard.writer import SummaryWriter

from pt.models.model_factory import ModelFactory
from pt.preprocessing.preprocess_json import load_datalist, resolve_datalist


class MammoLearner:
    def __init__(
        self,
        dataset_root: str,
        datalist_prefix: str,
        conf: dict[str, Any],
        datalist: str,
        aggregation_epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 64,
        architecture: str = "resnet",
        train_datalist: list[dict[str, str | int]] | None = None,
        valid_datalist: list[dict[str, str | int]] | None = None,
        run_name: str = "default",
    ):

        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point
        self.aggregation_epochs: int = aggregation_epochs
        self.batch_size: int = batch_size
        self.best_metric: float = 0.0
        self.datalist: str = datalist
        self.datalist_prefix: str = datalist_prefix
        self.dataset_root: str = dataset_root
        self.lr: float = lr
        self.num_classes: int = 0
        self.run = None
        # Epoch counter
        self.acc_values: list[float] = []
        self.arch: str = architecture
        self.config: dict[str, Any] = conf
        self.epoch_global: int = 0
        self.log: Logger = getLogger(__name__)
        self.roc_values: list[float] = []
        self.run_name = run_name
        self.train_datalist = train_datalist
        self.valid_datalist = valid_datalist

        # The following objects will be build in `initialize()`
        self.criterion: CrossEntropyLoss
        self.device: device
        self.model: Module
        self.optimizer: Adam
        self.sched: OneCycleLR
        self.train_dataset: CacheDataset
        self.train_loader: DataLoader
        self.transform_train: Compose
        self.transform_valid: Compose
        self.valid_dataset: CacheDataset | None
        self.valid_loader: DataLoader | None
        self.writer: SummaryWriter

    def save_model(self, name: str = "local_model.safetensors"):
        model_dir: str = self.config["io_dirs"].get("save_model_dir")
        run_dir: str = join(model_dir, self.run_name)
        makedirs(run_dir, exist_ok=True)
        model_path: str = join(run_dir, name)
        save_model(self.model, model_path)

    def build_transforms(self):
        self.transform_train = Compose(
            [
                LoadImaged(keys=["image"]),
                Transposed(keys=["image"], indices=[2, 0, 1]),
                RandRotated(keys=["image"], range_x=pi / 12, prob=0.5, keep_size=True),
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
                RandZoomd(
                    keys=["image"],
                    min_zoom=0.9,
                    max_zoom=1.1,
                    prob=0.5,
                    keep_size=True,
                ),
                CastToTyped(keys=["image"], dtype=float32),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        # 2. Transformações de Validação
        self.transform_valid = Compose(
            [
                LoadImaged(keys=["image"]),
                Transposed(keys=["image"], indices=[2, 0, 1]),
                CastToTyped(keys=["image"], dtype=float32),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

    def build_dataloaders(self):
        # Note, do not change this syntax. The data list filename is given by the system.
        datalist_file: str = self.datalist_prefix
        if not isfile(datalist_file):
            print(f"{datalist_file} does not exist!")

        train_datalist = (
            load_datalist(
                datalist_file,
                data_list_key="train",  # do not change this key name
                base_dir=self.dataset_root,
            )
            if self.train_datalist is None
            else resolve_datalist(self.train_datalist, self.dataset_root)
        )
        if not train_datalist:
            raise ValueError("No training images were found in the selected manifest")

        val_datalist = (
            load_datalist(
                datalist_file,
                data_list_key="test",
                base_dir=self.dataset_root,
            )
            if self.valid_datalist is None
            else resolve_datalist(self.valid_datalist, self.dataset_root)
        )

        num_workers: int = self.config["dataloaders"].get("num_workers", 4)
        cache_rate: int = self.config["dataloaders"].get("cache_rate", 1.0)

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
            num_workers=num_workers,
        )
        print(f"Training set: {len(train_datalist)} entries")

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
            print(f"Validation set: {len(val_datalist)} entries")
        else:
            self.valid_dataset = None
            self.valid_loader = None
            print("Use no validation set")

    def build_model(self):
        self.model = ModelFactory.create_model(
            architecture=self.arch, num_classes=self.num_classes
        )

    def build_optimizer(self):
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
        )

    def initialize(self):

        log_dir = self.config["io_dirs"].get("runs_dir")
        self.writer = SummaryWriter(
            log_dir=join(log_dir, self.run_name) if log_dir else None
        )

        layout: dict[str, dict[str, list[list[str] | str]]] = {
            "Analysis": {
                "loss": ["Multiline", ["train_loss", "val_loss"]],
                "accuracy": ["Multiline", ["train_acc", "val_acc"]],
            },
        }
        self.writer.add_custom_scalars(layout)

        self.build_transforms()

        self.num_classes: int = self.config["hyperparameters"].get("num_classes", 2)
        self.device = device("cuda:0" if is_available() else "cpu")

        self.build_dataloaders()

        self.build_model()

        self.model = self.model.to(self.device)
        self.build_optimizer()

        self.criterion = CrossEntropyLoss()

        self.criterion = self.criterion.to(self.device)

        # Set up one-cycle learning rate scheduler
        self.sched = OneCycleLR(
            self.optimizer,
            self.lr,
            epochs=self.aggregation_epochs,
            steps_per_epoch=len(self.train_loader),
        )

        print("Finished initializing")

    def get_lr(self, optimizer: Adam) -> float:
        for param_group in optimizer.param_groups:
            return param_group["lr"]
        return self.lr

    def train(self, train_loader: DataLoader) -> None:

        for epoch in range(self.aggregation_epochs):
            self.model.train()
            self.epoch_global = epoch + 1
            lrs: list[float] = []
            print(
                f"Local epoch: {epoch + 1}/{self.aggregation_epochs} (lr={self.lr})",
            )
            avg_loss: float = 0.0
            correct, total = 0, 0
            for batch_data in train_loader:
                inputs, labels = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )

                # Gradient Clipping for VGG-16
                if self.arch == "vgg":
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                # att, raw, outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                # Record & update learning rate
                lrs.append(self.get_lr(self.optimizer))
                self.sched.step()
                avg_loss += loss.item()

                _, _pred_label = max_torch(outputs.data, 1)
                _labels = batch_data["label"].to(self.device)
                total += inputs.data.size()[0]
                correct += (_pred_label == _labels.data).sum().item()

            self.writer.add_scalar("lr", self.get_lr(self.optimizer), epoch + 1)

            self.writer.add_scalar(
                "train_loss", avg_loss / len(train_loader), self.epoch_global
            )

            self.writer.add_scalar(
                "train_acc", correct / float(total), self.epoch_global
            )

            acc, kappa, roc = self.local_valid(self.valid_loader)
            if acc is not None:
                if len(self.acc_values) == 0 or acc >= max(self.acc_values):
                    self.save_model()
                self.acc_values.append(acc)
                self.writer.add_scalar("val_acc", acc, self.epoch_global)
            if kappa is not None:
                self.writer.add_scalar("val_kappa", kappa, self.epoch_global)
            if roc is not None:
                self.roc_values.append(roc)

    def local_valid(
        self,
        valid_loader: DataLoader | None,
        is_final: bool = False,
        fold: int | None = None,
    ) -> tuple[float | None, float | None, float | None]:
        if not valid_loader:
            return (None, None, None)
        self.model.eval()
        return_probs: list[dict[str, list[float] | str]] = []
        labels: list[int] = []
        pred_labels: list[int] = []
        l_probs: list[float] = []
        val_avg_loss: float = 0.0
        with no_grad():
            correct, total = 0, 0
            for batch_data in valid_loader:
                inputs, lbls = (
                    batch_data["image"].to(self.device),
                    batch_data["label"].to(self.device),
                )

                outputs = self.model(inputs)

                # Find the Loss
                validation_loss = self.criterion(outputs, lbls)
                val_avg_loss += validation_loss.item()
                outputs_soft: Tensor = softmax(outputs, dim=1)
                probs = outputs_soft.detach().cpu().numpy()
                _, _pred_label = max_torch(outputs_soft.data, 1)
                _labels = batch_data["label"].to(self.device)
                total += inputs.data.size()[0]
                correct += (_pred_label == _labels.data).sum().item()
                labels.extend(_labels.detach().cpu().numpy())
                pred_labels.extend(_pred_label.detach().cpu().numpy())

                # make json serializable
                for _img_file, _probs, lbl in zip(
                    batch_data["image"].meta["filename_or_obj"],
                    probs,
                    batch_data["label"],
                ):
                    p: list[float] = [float(p) for p in _probs]
                    return_probs.append(
                        {
                            "image": basename(_img_file),
                            "probs": p,
                            "label": lbl,
                        }
                    )
                    l_probs.append(p[1])  # probs da classe positiva

            self.writer.add_scalar(
                "val_loss", (val_avg_loss / len(valid_loader)), self.epoch_global
            )

            acc: float = correct / float(total)
            assert len(labels) == total
            assert len(pred_labels) == total
            matrix = confusion_matrix(
                labels, pred_labels, labels=list(range(self.num_classes))
            )
            print("### eval report ###")
            roc_auc: float | None = None
            if self.num_classes == 2:
                if len(set(labels)) == 2:
                    roc_auc = roc_auc_score(labels, l_probs)
                f1 = f1_score(labels, pred_labels)
                print(f"ROC Score: {roc_auc if roc_auc is not None else 'undefined'}")
                print(f"F1-Score: {f1}")

            mcc: float = matthews_corrcoef(labels, pred_labels)
            kappa: float = cohen_kappa_score(labels, pred_labels, weights="linear")

            print(f"ACC: {acc}")
            print(f"MCC: {mcc}")
            print(f"Cohen Kappa Score: {kappa}")
            print(matrix)
            print("###################")

            if is_final:
                if self.num_classes == 2:
                    # ROC curve
                    fold_in_title: str = f" for fold {fold}" if fold is not None else ""
                    fpr, tpr, _ = roc_curve(labels, l_probs)
                    plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                    xlim([0, 1])
                    ylim([0, 1])
                    xlabel("False Positive Rate")
                    ylabel("True Positive Rate")
                    title(f"ROC Curve{fold_in_title}")
                    legend()
                    show()
                    print(f"ROC VALUES: {self.roc_values}")
                    print(f"ACC VALUES: {self.acc_values}")

                # CONFUSION MATRIX
                row_totals = matrix.sum(axis=1, keepdims=True)
                cm_norm = np.divide(
                    matrix.astype("float"),
                    row_totals,
                    out=np.zeros_like(matrix, dtype=float),
                    where=row_totals != 0,
                )

                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm_norm, display_labels=range(self.num_classes)
                )
                disp.plot()

            return acc, kappa, roc_auc
