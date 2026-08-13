import torch
from torch import nn

from model.attention import MultiHeadAttention
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import GELU


class EncoderBlock(nn.Module):
    """ Encoder block 

    A simplefied verison of what is implemented in "Attention is all you need".
    The simplification is in the use of only ONE attention head. 

    If the dimensionality of the input vector (dx) is not equal to the dimensionality
    of the layer (d_layer), the input is not added to the output of the attention 
    layer before normalization.

    Parameters:
        - dx (int): Dimensionality of the embedding
        - h (int):  Number of heads
        - dh (int): Dimensionality per head
        - d_layer (int):    Output dimensionality of the encoder layer. The final forward
                            layer goes from h*dh to d_layer
    """
    def __init__(self, dx: int, h: int, dh: int, d_layer: int, mask=None, act=GELU):
        super().__init__()
        self.dx = dx
        self.h = h
        self.dh = dh
        self.d_layer = d_layer 
        self.mask = mask
        self.act = act

        self.mh_attention_layer = MultiHeadAttention(
                h = self.h,
                dx = self.dx,
                dy = self.dx,
                dk = self.dh,
                dv = self.dh,
                mask = self.mask,
                act = self.act
            )
        self.mh_attention_norm_layer = LayerNorm(self.h*self.dh)

        self.feed_forward_layer = nn.Sequential(
                nn.Linear(self.h*self.dh, self.d_layer, bias=False),
                GELU(),
            )
        self.ff_norm_layer = LayerNorm(self.d_layer)


    def forward(self, x):
        attention = self.mh_attention_layer.forward(x, x)
        normalized_attention = torch.zeros_like(attention)
        if self.dx==self.d_layer:
            normalized_attention = self.mh_attention_norm_layer(attention + x)
        else:
            normalized_attention = self.mh_attention_norm_layer(attention)
        feed_forward = self.feed_forward_layer(normalized_attention)
        
        return self.ff_norm_layer(feed_forward + normalized_attention)


class EncoderStack(nn.Module):
    """ A stack of encoders

    The dimensionality in the first encoder in the stack goes from 
    dx -> d_layer. All following go from d_layer -> d_layer.
    
    Parameters:
        - n (int):  Number of encoders to stack
        - dx (int): Dimensionality of the embedding
        - h (int):  Number of heads
        - dh (int): Dimensionality per head
        - d_layer (int):    Output dimensionality of the encoder layer. The final forward
                            layer goes from h*dh to d_layer
    """
    def __init__(self, n, dx, h, dh, d_layer, mask=None, act=GELU):
        super().__init__()
        self.n = n
        self.dx = dx
        self.h = h
        self.dh = dh
        self.d_layer = d_layer 
        self.mask = mask
        self.act = act

        self.first_encoder = EncoderBlock(
                dx = self.dx,
                h = self.h,
                dh = self.dh,
                d_layer = self.d_layer,
                mask = self.mask,
                act = self.act
            )
        self.remaining_encoder_list = nn.ModuleList([
                EncoderBlock(
                    dx = self.d_layer,
                    h = self.h,
                    dh = self.dh,
                    d_layer = self.d_layer,
                    mask = self.mask,
                    act = self.act
                ) for i in range(1, self.n)
            ]) 

    def forward(self, x):
        x = self.first_encoder(x)
        for e in self.remaining_encoder_list:
            x = e(x)
        return x
