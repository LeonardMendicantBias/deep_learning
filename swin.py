from typing import Callable, List, Tuple
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim
from torchvision.ops import StochasticDepth, Permute, MLP
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

import lightning as pl
import torchmetrics as metrics

import attention


class PatchMerging(nn.Module):

    def __init__(self,
            embed_dim: int,
            norm_layer: Callable[..., nn.Module]=nn.LayerNorm
        ) -> None:
        super().__init__()
        self.reduction = nn.Linear(4*embed_dim, 2*embed_dim, bias=False)
        self.norm = norm_layer(2*embed_dim)

    def forward(self, x: Tensor):
        _, H, W, _ = x.shape
        x = F.pad(x, (0, 0, 0, W%2, 0, H%2))  # pad to an even number
        x_0 = x[..., 0::2, 0::2, :]
        x_1 = x[..., 1::2, 0::2, :]
        x_2 = x[..., 0::2, 1::2, :]
        x_3 = x[..., 1::2, 1::2, :]
        x = torch.cat([x_0, x_1, x_2, x_3], dim=-1)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class SwinBlock(nn.Module):

    def __init__(self, 
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module]=nn.LayerNorm,
        attn_layer: Callable[..., nn.Module]=attention.ShiftWindowMHA,
    ) -> None:
        super().__init__()

        self.attn_norm = norm_layer(dim)
        self.attn = attn_layer(
            window_size,
            shift_size,
            dim,
            num_heads,
            attention_dropout,
            dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.mlp_norm = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim*mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)
        
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn_norm(self.attn(x)))
        x = x + self.stochastic_depth(self.mlp_norm(self.mlp(x)))
        return x
        

class SwinTokenizer(nn.Sequential):

    def __init__(self, embed_dim: int, patch_size: Tuple[int, int], norm_layer: Callable[..., nn.Module]):
        super().__init__(
            nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim)
        )


class SwinTransformer(nn.Module):

    def __init__(self,
        patch_size: Tuple[int, int],
        window_size: Tuple[int, int],
        depths: List[int],
        embed_dims: List[int],
        num_heads: List[int],
        attention_dropout: float, dropout: float,
        stochastic_depth_prob: int,
        # tokenizer: Callable[..., nn.Module]=None,
        norm_layer: Callable[..., nn.Module]=None,
        attn_block: Callable[..., nn.Module]=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if attn_block is None:
            attn_block = SwinBlock

        # split image into non-overlapping patches
        self.tokenizer = nn.Sequential(
            nn.Conv2d(3, embed_dims[0], kernel_size=patch_size, stride=patch_size),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dims[0])
        )
        # self.tokenizer = tokenizer(embed_dims[0], patch_size, norm_layer)

        self.layers = nn.ModuleList([])
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module]=[]
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks-1)
                stage.append(
                    SwinBlock(
                        embed_dims[i_stage],
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer%2==0 else s//2 for s in window_size],
                        mlp_ratio=4.0,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer
                    )
                )
                stage_block_id += 1
            # add patch merging layer
            if i_stage < (len(depths)-1):
                stage.append(PatchMerging(embed_dims[i_stage]))
            self.layers.append(nn.Sequential(*stage))

    def forward(self, x):
        x = self.tokenizer(x)
        features = [x]
        # print(x.shape)
        for layer in self.layers:
            x = layer(x)
            # print(x.shape)
            features.append(x)
        return features
    
    @classmethod
    def arda_swin(cls):
        # double the # of heads or # of dimensions
        return cls(
            patch_size=[4, 4],
            window_size=[8, 8],
            depths=[2, 2, 4, 2, 2],
            embed_dims=[96, 96, 192, 192, 384],
            num_heads= [ 3,  6,   6,  12,  12],
            attention_dropout=0.1, dropout=0.1,
            stochastic_depth_prob=0.2
        )
    
    @classmethod
    def swin_t(cls, is_pretrain=True):
        swin = cls(
            patch_size=[4, 4],
            window_size=[8, 8],
            depths=[2, 2, 6, 2],
            embed_dims=[96, 192, 384, 768],
            num_heads=[3, 6, 12, 24],
            attention_dropout=0.1, dropout=0.1,
            stochastic_depth_prob=0.2
        )
        if not is_pretrain:
            return swin
        
        pretrain = swin_v2_t(Swin_V2_T_Weights.IMAGENET1K_V1)
        layers = list(pretrain.children())
        layers_ = list(swin.children())
        
        # for base, target in zip(swin.head.children(), layers[1:]):
        #     base.load_state_dict(target.state_dict())

        for base, target in zip(layers_[0], list(layers[0].children())[0]):
            # print(base, target)
            base.load_state_dict(target.state_dict())

        for base, target in zip(layers_[1][:-1], list(layers[0].children())[2::2]):
            layer = list(base.children())[-1]
            layer.reduction.load_state_dict(target.reduction.state_dict())
            layer.norm.load_state_dict(target.norm.state_dict())

        for base, target in zip(layers_[1], list(layers[0].children())[1::2]):
            # base.load_state_dict(target.state_dict())
            for b, t in zip(list(base.children())[:-1], list(target.children())):
                dim = t.attn.qkv.weight.shape[0]
                d = dim // 3
                b.attn.query.weight = nn.Parameter(t.attn.qkv.weight[0:d].clone())
                b.attn.key.weight = nn.Parameter(t.attn.qkv.weight[d:d*2].clone())
                b.attn.value.weight = nn.Parameter(t.attn.qkv.weight[-d:].clone())
                b.attn.query.bias = nn.Parameter(t.attn.qkv.bias[0:d].clone())
                b.attn.key.bias = nn.Parameter(t.attn.qkv.bias[d:d*2].clone())
                b.attn.value.bias = nn.Parameter(t.attn.qkv.bias[-d:].clone())
                # print(t.attn.query)
                b.attn_norm.load_state_dict(t.norm1.state_dict())
                b.mlp_norm.load_state_dict(t.norm2.state_dict())
                b.mlp.load_state_dict(t.mlp.state_dict())
                # b.attn.cpb_mlp.load_state_dict(t.attn.cpb_mlp.state_dict())
                b.attn.linear.load_state_dict(t.attn.proj.state_dict())
                b.attn.cpb_mlp.load_state_dict(t.attn.cpb_mlp.state_dict())
                b.attn.logit_scale = nn.Parameter(t.attn.logit_scale.clone())
                b.attn.relative_position_index = t.attn.relative_position_index.clone()
                b.attn.relative_coords_table = t.attn.relative_coords_table.clone()
        
        return swin


class LitSwin(pl.LightningModule):
    
    def __init__(self,
        patch_size: Tuple[int, int],
        window_size: Tuple[int, int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        attention_dropout: float, dropout: float,
        stochastic_depth_prob: int,
    ):
        super().__init__()

        self.swin = SwinTransformer.swin_t()
        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.head = nn.Sequential(
            nn.LayerNorm(num_features, eps=1e-5),
            Permute([0, 3, 1, 2]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(num_features, 1000),
        )
        
        self.accuracy = metrics.Accuracy(task="multiclass", num_classes=1000)

    def forward(self, x):
        return self.head(self.swin(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        features = self.swin(x)
        logits = self.head(features)
        
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        features = self.swin(x)
        logits = self.head(features)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)

        self.accuracy.update(logits, y)
        # self.log('valid_acc', self.accuracy, on_step=True, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.accuracy.compute())
        self.accuracy.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # def _transfer_parameter(self, base, target):


    @classmethod
    def swin_t(cls):
        model = swin_v2_t(Swin_V2_T_Weights.IMAGENET1K_V1)
        swin = cls(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=[8, 8],
            attention_dropout=0.1, dropout=0.1,
            stochastic_depth_prob=0.2
        )
        # print(swin)
        
        layers = list(model.children())
        layers_ = list(swin.swin.children())

        for base, target in zip(swin.head.children(), layers[1:]):
            base.load_state_dict(target.state_dict())

        return swin
