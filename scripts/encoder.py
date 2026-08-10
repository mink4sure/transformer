import numpy as np
import torch
from torch import nn

from attention import Attention
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import GELU


class Encoder(nn.Module):
    """ Encoder Layer

    A simplefied verison of what is implemented in "Attention is all you need".
    The simplification is in the use of only ONE attention head. 

    If the dimensionality of the input vector (dx) is not equal to the dimensionality
    of the model (d_model), the input is not added to the output of the attention 
    layer before normalization.

    Input:
        - dx (int): Dimensionality of the embedding
        - d_model (int): Dimensionality of the query, key and value tensors
                         in the self-attention layer

    """
    def __init__(self, dx: int, d_model: int, mask=None, act=GELU):
        super().__init__()
        self.dx = dx
        self.d_model = d_model

        self.attention_layer = Attention(dx=dx, dy=dx, dk=d_model, dv=d_model, mask=mask, act=act)
        self.attention_norm_layer = LayerNorm(self.d_model)

        self.feed_forward_layer = nn.Sequential(
                nn.Linear(self.d_model, self.d_model, bias=False),
                GELU(),
            )
        self.ff_norm_layer = LayerNorm(self.d_model)


    def forward(self, x):
        attention = self.attention_layer.forward(x, x)
        normalized_attention = torch.zeros_like(attention)
        if self.dx==self.d_model:
            normalized_attention = self.attention_norm_layer(attention + x)
        else:
            normalized_attention = self.attention_norm_layer(attention)
        feed_forward = self.feed_forward_layer(normalized_attention)
        
        return self.ff_norm_layer(feed_forward + normalized_attention)


class EncoderStack(nn.Module):
    """ A stack of encoders
    """
    def __init__(self, n, dx, d_model, mask=None, act=GELU):
        super().__init__()
        self.first_encoder = Encoder(dx, d_model, mask, act)
        self.remaining_encoder_list = nn.ModuleList(
                [Encoder(d_model, d_model, mask, act) for i in range(1, n)]
            ) 

    def forward(self, x):
        x = self.first_encoder(x)
        for e in self.remaining_encoder_list:
            x = e(x)
        return x
