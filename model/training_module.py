"""
PyTorch Lightning modules for training and evaluation.

Implements LitModule: generic Lightning wrapper with train/val/test steps,
optimizer & scheduler configuration and metric logging.
"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from typing import Optional, Dict, Any


class LitModule(LightningModule):
    """
    Generic PyTorch Lightning training module.

    Wraps any ``nn.Module`` with:
    - Cross-entropy loss
    - Accuracy + F1 metrics (train / val / test)
    - Configurable optimizer & LR scheduler
    """

    def __init__(self, net: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 lr_scheduler: Optional[Any] = None,
                 parameters: Optional[Dict] = None):
        super().__init__()
        self.net = net
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        params = parameters or {}
        self.num_classes = params.get('num_classes', 6)

        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.val_acc   = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.test_acc  = Accuracy(task='multiclass', num_classes=self.num_classes)
        self.train_f1  = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        self.val_f1    = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_f1   = F1Score(task='multiclass', num_classes=self.num_classes, average='macro')
        self.test_cm   = ConfusionMatrix(task='multiclass', num_classes=self.num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


    def training_step(self, batch, batch_idx):
        embeddings, labels = batch['embeddings'], batch['label']
        logits = self(embeddings)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        embeddings, labels = batch['embeddings'], batch['label']
        logits = self(embeddings)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True)
        return loss


    def test_step(self, batch, batch_idx):
        embeddings, labels = batch['embeddings'], batch['label']
        logits = self(embeddings)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_cm(preds, labels)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        if self._optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            optimizer = self._optimizer
        if self._lr_scheduler is not None:
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': self._lr_scheduler,
                                     'interval': 'epoch'}}
        return optimizer
