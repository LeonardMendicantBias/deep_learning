import math
from functools import partial
from typing import Tuple, Callable, List, Optional

import torchmetrics

import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.ops import StochasticDepth, Permute, MLP
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.datasets import ImageNet, CocoDetection

import lightning.pytorch as pl
import torchmetrics

import tokenizer
from timm.scheduler.cosine_lr import CosineLRScheduler

    
class LitPreTrainSwin(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.swin = swin_v2_t()
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000)

    def forward(self, x):
        return self.swin(x)

    def training_step(self, batch, batch_idx):
        schs = self.lr_schedulers()
        for sch in schs:
            sch.step()

        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

        self.accuracy.update(logits, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.accuracy.compute())
        self.accuracy.reset()

    def configure_optimizers(self):
        # optimizer = optim.AdamW(4e-5, weight_decay=1e-8)
        lr = 5e-4
        optimizer = optim.AdamW(self.parameters(), lr, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
        # warm_up = optim.lr_scheduler.LinearLR(optimizer,
        #     start_factor=0, end_factor=lr, last_epoch=21, verbose=True
        # )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, gamma=0.9)

        # n_iter_per_epoch = 1251
        # num_steps = int(300*n_iter_per_epoch)
        # warmup_steps = int(20*n_iter_per_epoch)
        # lr_scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=(num_steps - warmup_steps),
        #     cycle_mul=1.,
        #     lr_min=5e-6,
        #     warmup_lr_init=5e-7,
        #     warmup_t=warmup_steps,
        #     cycle_limit=1,
        #     t_in_epochs=False,
        #     warmup_prefix=True,
        # )
        # return ([optimizer], [lr_scheduler])
        return optimizer


if __name__ == '__main__':
    train_ds = ImageNet('/mnt/data/dataset/ImageNet/', split='train', transform=Swin_V2_T_Weights.DEFAULT.transforms())
    train_loader = utils.data.DataLoader(train_ds, num_workers=4, batch_size=1024, shuffle=True, pin_memory=True)
    
    val_ds = ImageNet('/mnt/data/dataset/ImageNet/', split='val', transform=Swin_V2_T_Weights.DEFAULT.transforms())
    val_loader = utils.data.DataLoader(val_ds, num_workers=4, batch_size=1024, shuffle=True, pin_memory=True)

    model = LitPreTrainSwin()
    
    trainer = pl.Trainer(
        precision='16-mixed', accelerator="gpu",
        gradient_clip_val=5.0, gradient_clip_algorithm="norm",
        limit_train_batches=1,
        # limit_train,val,test,predict}_batches
        num_sanity_val_steps=0,
        # limit_train_batches=100, max_epochs=1,
        # devices=1, num_nodes=1
    )
    # trainer.validate(model, val_loader, verbose=True)
    trainer.fit(model, train_loader, val_loader)

