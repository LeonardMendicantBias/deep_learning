from typing import Tuple, Callable, List, Optional

import torch
from torch import optim, nn, utils, Tensor
from torchvision.ops import StochasticDepth, Permute, MLP


class ImagePartition(nn.Sequential):
            
    def __init__(self, embed_dim: int, patch_size: List[int]):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if len(patch_size) != 2:
            raise ValueError('patch_size must be a tuple/list of length 2')

        layers = [
            nn.Conv2d(
                3, embed_dim,
                kernel_size=(patch_size[0], patch_size[1]),
                stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]),  # B C W H -> B W H C
            nn.LayerNorm(embed_dim),
        ]
        super().__init__(*layers)
