import math
from functools import partial
from typing import Tuple, Callable, List, Optional

import torchmetrics

import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.ops import StochasticDepth, Permute, MLP
from torchvision.models import swin_v2_t, Swin_V2_T_Weights

import lightning.pytorch as pl
import torchmetrics

import tokenizer


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


class PatchMerging(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.reduction = nn.Conv2d(
            dim, 2*dim,
            (2, 2), (2, 2), bias=False
        )
        self.norm = nn.LayerNorm(2 * dim)  # difference

    def forward(self, x: Tensor):
        _, H, W, _ = x.shape
        x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x = x.permute((0, 3, 1, 2))
        
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = x.permute((0, 2, 3, 1))
        x = self.norm(x)
        return x
    
    # def forward(self, x: Tensor):
    #     _, H, W, _ = x.shape
    #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    #     print(x.shape)
    #     x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
    #     x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
    #     x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
    #     x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
    #     x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
    #     print(x.shape)
        
    #     x = self.reduction(x)  # ... H/2 W/2 2*C
    #     x = self.norm(x)
    #     return x


class SwinAttention(nn.Module):

    def __init__(self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        attention_dropout: float,
        dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.query = self.build_query()
        self.key = self.build_key()
        self.value = self.build_value()
        self.linear = self.build_linear()

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )

    def build_query(self):
        return nn.Linear(self.dim, self.dim)

    def build_key(self):
        return nn.Linear(self.dim, self.dim)
    
    def build_value(self):
        return nn.Linear(self.dim, self.dim)
    
    def build_linear(self):
        return nn.Linear(self.dim, self.dim)

    def _pre_attention(self, x):
        B, H, W, C = x.size()
        window_size = self.window_size
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, pad_H, pad_W, _ = x.shape

        shift_size = self.shift_size.copy()
        # If window size is larger than feature size, there is no need to shift window
        if self.window_size[0] >= pad_H:
            shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            shift_size[1] = 0
        # cyclic shift
        # if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        # print(x.shape)
        # partition windows
        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
        # print('num_windows', num_windows)
        x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        # print(x.shape)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)
        # print(x.shape)
        return x, pad_H, pad_W, num_windows

    def _logits(self, q: Tensor, k: Tensor) -> Tensor:
        return F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)

    def _attention(self): ...
    
    def _post_attention(self, attn):
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
        return attn + self.get_relative_position_bias()
    
    # def _linear(self, v: Tensor) -> Tensor:
    #     return self.linear(v)

    def define_relative_position_bias_table(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor):
        B, H, W, C = x.size()
        # print(B, H, W, C)
        # window_size = self.window_size
        # pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        # pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        # x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        # _, pad_H, pad_W, _ = x.shape

        # # If window size is larger than feature size, there is no need to shift window
        # if self.window_size[0] >= pad_H:
        #     shift_size[0] = 0
        # if self.window_size[1] >= pad_W:
        #     shift_size[1] = 0
        # # cyclic shift
        # if sum(shift_size) > 0:
        #     x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        # # partition windows
        # num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
        # x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        # x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
        
        x, pad_H, pad_W, num_windows = self._pre_attention(x)
        shift_size = self.shift_size.copy()
        if self.window_size[0] >= pad_H:
            shift_size[0] = 0
        if self.window_size[1] >= pad_W:
            shift_size[1] = 0
        q, k, v = self.query(x), self.key(x), self.value(x)
        q = q.reshape(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = self._logits(q, k) # F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = self._post_attention(attn)
        
        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -shift_size[0]), (-shift_size[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -shift_size[1]), (-shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // self.window_size[0], self.window_size[0], pad_W // self.window_size[1], self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            print('abc', attn.shape)
            attn = attn.view(x.size(0) // num_windows, num_windows, self.num_heads, x.size(1), x.size(1))
            print(attn.shape, attn_mask.shape)
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)

        x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
        x = self.linear(x)
        x = self.dropout(x)

        # reverse windows
        x = x.view(B, pad_H // self.window_size[0], pad_W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

        # reverse cyclic shift
        if sum(shift_size) > 0:
            x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

        # unpad features
        x = x[:, :H, :W, :].contiguous()
        return x


class SwinBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        stochastic_depth_prob: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = SwinAttention(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x
    
class SwinTransformer(nn.Module):

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

        # split image into non-overlapping patches
        self.tokenizer = tokenizer.ImagePartition(embed_dim, patch_size)

        self.layers = nn.ModuleList([])
        total_stage_blocks = sum(depths)
        stage_block_id = 0

        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    SwinBlock(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=4.0,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        # norm_layer=partial(nn.LayerNorm, eps=1e-5),
                    )
                )
                stage_block_id += 1
            self.layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                self.layers.append(PatchMerging(dim))
        # self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)
        return x

class ClassificationHead(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, eps=1e-5)
        self.permute = Permute([0, 3, 1, 2])  # B H W C -> B C H W
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.head(x)
        return logits

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

        self.swin = SwinTransformer(
            patch_size, window_size,
            embed_dim, depths, num_heads,
            attention_dropout, dropout,
            stochastic_depth_prob
        )
        self.head = ClassificationHead(embed_dim * 2 ** (len(depths) - 1), 1000)
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1000)

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

        for base, target in zip(layers_[0], list(layers[0].children())[0]):
            base.load_state_dict(target.state_dict())

        for base, target in zip(layers_[1][1::2], list(layers[0].children())[2::2]):
            base.reduction.load_state_dict(target.reduction.state_dict())
            base.norm.load_state_dict(target.norm.state_dict())

        for base, target in zip(layers_[1][0::2], list(layers[0].children())[1::2]):
            # base.load_state_dict(target.state_dict())
            for b, t in zip(list(base.children()), list(target.children())):
                dim = t.attn.qkv.weight.shape[0]
                d = dim // 3
                b.attn.query.weight = nn.Parameter(t.attn.qkv.weight[0:d].clone())
                b.attn.key.weight = nn.Parameter(t.attn.qkv.weight[d:d*2].clone())
                b.attn.value.weight = nn.Parameter(t.attn.qkv.weight[-d:].clone())
                b.attn.query.bias = nn.Parameter(t.attn.qkv.bias[0:d].clone())
                b.attn.key.bias = nn.Parameter(t.attn.qkv.bias[d:d*2].clone())
                b.attn.value.bias = nn.Parameter(t.attn.qkv.bias[-d:].clone())
                # print(t.attn.query)
                b.norm1.load_state_dict(t.norm1.state_dict())
                b.norm2.load_state_dict(t.norm2.state_dict())
                b.mlp.load_state_dict(t.mlp.state_dict())
                # b.attn.cpb_mlp.load_state_dict(t.attn.cpb_mlp.state_dict())
                b.attn.linear.load_state_dict(t.attn.proj.state_dict())
                b.attn.cpb_mlp.load_state_dict(t.attn.cpb_mlp.state_dict())
                b.attn.logit_scale = nn.Parameter(t.attn.logit_scale.clone())
                b.attn.relative_position_index = t.attn.relative_position_index.clone()
                b.attn.relative_coords_table = t.attn.relative_coords_table.clone()

        return swin

    @classmethod
    def s(cls):
        return cls(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 2, 2, 2],
            num_heads=[3, 6, 12, 24, 48],
            window_size=[8, 8],
            attention_dropout=0.1, dropout=0.1,
            stochastic_depth_prob=0.1
        )
