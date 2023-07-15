# %%
import torch
torch.set_float32_matmul_precision('high')
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.ops import StochasticDepth, Permute, MLP

from torchvision.datasets import ImageNet, CocoDetection

import lightning.pytorch as pl
import torchmetrics

import net


if __name__ == '__main__':
    # val_ds = ImageNet('/mnt/data/dataset/ImageNet/', split='val', transform=Swin_V2_T_Weights.DEFAULT.transforms())
    val_ds = ImageNet('~/ImageNet/', split='val', transform=Swin_V2_T_Weights.DEFAULT.transforms())
    val_loader = utils.data.DataLoader(val_ds, num_workers=4, batch_size=512, shuffle=False, pin_memory=True)

    # model = swin_v2_t(Swin_V2_T_Weights.IMAGENET1K_V1)

    model = net.LitSwin.swin_t()
    # model = net.LitSwinTorch()
    trainer = pl.Trainer(
        precision='16-mixed', accelerator="gpu",# strategy='ddp',
        limit_train_batches=100, max_epochs=1,
        devices=1, num_nodes=1
    )
    trainer.validate(model, val_loader, verbose=True)

