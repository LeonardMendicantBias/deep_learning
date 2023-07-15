import torch
from torch import optim, nn, utils, Tensor

from typing import List


class MultiHeadAttention(nn.Module):

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