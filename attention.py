import math
from typing import Union, Tuple, List

import torch
from torch import nn, Tensor, BoolTensor
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float,
        dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = self.build_query()
        self.key = self.build_key()
        self.value = self.build_value()
        self.linear = self.build_linear()

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dropout = nn.Dropout(dropout)

    def build_query(self):
        return nn.Linear(self.embed_dim, self.embed_dim)

    def build_key(self):
        return nn.Linear(self.embed_dim, self.embed_dim)
    
    def build_value(self):
        return nn.Linear(self.embed_dim, self.embed_dim)
    
    def build_linear(self):
        return nn.Linear(self.embed_dim, self.embed_dim)
    
    def _headify(self, x: Tensor):  # split the last dim
        B, T, D = x.size()
        x = x.view(B, T, self.num_heads, D//self.num_heads).transpose(1, 2)
        return x  # (B, H, T, D)
    
    def _deheadify(self, x: Tensor):  # merge the last two dims
        B, H, T, D = x.shape
        x = x.transpose(1, 2).reshape(B, T, H*D)
        return x
    
    def _calculate_logits(self, q: Tensor, k: Tensor):
        return (q @ k.mT) / math.sqrt(self.embed_dim)
    
    def _attention(self, logits: Tensor, mask: BoolTensor):
        if mask is not None:
            logits = logits.masked_fill(mask, float('-inf'))
        return F.softmax(logits, dim=-1)
    
    def _aggregate(self, scores: Tensor, v: Tensor):
        return scores @ v
    
    def forward(self, x: Tensor, mask: Tensor=None):
        q, k, v = self.query(x), self.key(x), self.value(x)
        q, k, v = self._headify(q), self._headify(k), self._headify(v)

        logits = self._calculate_logits(q, k)
        scores = self._attention(logits, mask)
        scores = self.attention_dropout(scores)

        o = self._aggregate(scores, v)
        o = self._deheadify(o)

        z = self.linear(o)
        return self.dropout(z)
    
def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


class ShiftWindowMHA(MultiHeadAttention):

    def __init__(self, 
            window_size: Union[int, Tuple[int, int]],
            shift_size: Union[int, Tuple[int, int]],
            embed_dim: int, num_heads: int, attention_dropout: float, dropout: float
        ):
        super().__init__(embed_dim, num_heads, attention_dropout, dropout)
        self.window_size = torch.tensor(window_size, dtype=torch.int)
        self.shift_size = torch.tensor(shift_size, dtype=torch.int)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_heads, bias=False)
        )
        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def build_query(self):
        return nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def build_key(self):
        return nn.Linear(self.embed_dim, self.embed_dim, bias=False)

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

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias
    
    def _calculate_logits(self, q: Tensor, k: Tensor) -> Tensor:
        '''
            Updated logit calculation that uses cosine instead of dot-product
        '''
        logit_scale = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        # relative_position_bias = 
        return attn * logit_scale + self.get_relative_position_bias()
    
    def _attention(self, logits: Tensor, mask: Tensor):
        if mask is not None:
            # print(logits.shape)
            logits = logits.view(logits.size(0) // 4, 4, self.num_heads, logits.size(2), logits.size(2))
            # print(logits.shape, mask.unsqueeze(1).shape)
            logits = logits.masked_fill(mask.unsqueeze(1).unsqueeze(0), float('-inf'))
            logits = logits.view(-1, self.num_heads, logits.size(-1), logits.size(-2))
        return F.softmax(logits, dim=-1)

    def _cyclic_shift(self, img: Tensor):
        size = torch.tensor([img.size(1), img.size(2)], dtype=torch.int)
        shift = torch.where(self.window_size >= size, 0, self.shift_size)
        shift_img = torch.roll(img, shifts=(-shift[0], -shift[1]), dims=(1, 2))
        
        if sum(shift) > 0:
            num_windows = (img.size(1) // self.window_size[0]) * (img.size(2) // self.window_size[1])
            # generate attention mask
            attn_mask = img.new_zeros((img.size(1), img.size(2)), dtype=torch.int)
            h_slices = ((0, -self.window_size[0]), (-self.window_size[0], -shift[0]), (-shift[0], None))
            w_slices = ((0, -self.window_size[1]), (-self.window_size[1], -shift[1]), (-shift[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(img.size(1) // self.window_size[0], self.window_size[0], img.size(2) // self.window_size[1], self.window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows.prod(), self.window_size[0] * self.window_size[1])
            attn_mask = attn_mask.unsqueeze(1) != attn_mask.unsqueeze(2)

        x = self._reshape(shift_img, attn_mask)
        
        img = torch.roll(x, shifts=(shift[0], shift[1]), dims=(1, 2))
        return img
    
    def _pad(self, img: Tensor):
        size = torch.tensor([img.size(1), img.size(2)], dtype=torch.int, device=img.device)
        pad = (self.window_size - size%self.window_size)%self.window_size
        pad_img = F.pad(img, (0, 0, 0, pad[1], 0, pad[0]))

        img = self._cyclic_shift(pad_img)

        return img[:, :size[0], :size[1], :]
    
    def _reshape(self, img: Tensor, mask: Tensor):
        B, H, W, D = img.size()

        w_s = self.window_size
        img = img.view(B, H//w_s[0], w_s[0], W//w_s[1], w_s[1], D)
        img = img.permute(0, 1, 3, 2, 4, 5)
        img = img.reshape(B*(H//w_s[0])*(W//w_s[1]), w_s.prod(), D)

        x = super().forward(img, mask)

        img = x.reshape(B, H//w_s[0], W//w_s[1], w_s[0], w_s[1], D)
        img = img.permute(0, 1, 3, 2, 4, 5)
        img = img.reshape(B, H, W, D)

        return img

    def forward(self, img: Tensor, mask: Tensor=None):
        x = self._pad(img)

        return x


class DeformableMHA(MultiHeadAttention):

    def __init__(self, num_levels: int, num_points: int, embed_dim: int, num_heads: int, attention_dropout: float, dropout: float):
        super().__init__(embed_dim, num_heads, attention_dropout, dropout)
        self.num_levels = num_levels

        self.im2col_step = 64

        self.sampling_offsets = nn.Linear(embed_dim, num_heads*num_levels*num_points*2)
        self.attetntion_weights = nn.Linear(embed_dim, num_heads*num_levels*num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, reference_points):
        pass
