from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import StochasticDepth, Permute, MLP

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
        x = F.pad(x, (0, 0, 0, W%2, 0, H%2))
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
        norm_layer: Callable[..., nn.Module]=nn.LayerNorm
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim, eps=1e-5)
        self.attn = attention.SwinAttention(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim, eps=1e-5)
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
        self.tokenizer = PatchMerging(embed_dim, patch_size)

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

    def forward(self, x):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)
        return x

# class SwinClassifier(nn.Module):
