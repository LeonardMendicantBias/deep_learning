from typing import Callable, List, Tuple
from functools import partial

from torch import nn, Tensor, optim
from torchvision.ops import StochasticDepth, Permute, MLP

import attention


class TransformBlock(nn.Module):

    def __init__(self,
            embed_dim: int,
            num_heads: int,
            dropout: float,
            attention_dropout: float,
            stochastic_depth_prob: float,
            attn_layer: Callable[..., nn.Module]=nn.LayerNorm,
            norm_layer: Callable[..., nn.Module]=nn.LayerNorm
        ):
        super().__init__()
        self.attn_norm = norm_layer(embed_dim)
        self.mlp_norm = norm_layer(embed_dim)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.mlp = MLP(embed_dim, [int(embed_dim*mlp_ratio), embed_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x
