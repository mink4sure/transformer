import torch
from torch import nn

from model.attention import MultiHeadAttention
from torch.nn.modules.activation import GELU


class DecoderBlock(nn.Module):
    """ A decoder block as described in 'Attention is all you need'.

    Parameters:
        - din:  Dimensionality of the previous layer
        - dx:   Dimensionality of the embedding
        - h:    Number of heads for each multi head attention layer
        - dh:   Dimensionality of each head
        - dout: Output dimensionality of the feed forward layer
        - mask: The mask used in the attention layer
        - act:  The activation function used in the attention layer

    The paper implements a model where h*dh=dout, however I won't let
    this be a necessary constraint.
    """
    def __init__(self, din, dx, h, dh, dout, mask=None, act=GELU):
        super().__init__()
        self.din = din
        self.dx = dx
        self.h = h
        self.dh = dh
        self.dout = dout
        self.mask = mask
        self.act = act

        self.mh_self_attention_layer = MultiHeadAttention(
                h = self.h,
                dx = self.din, # Self attention is performed on the output of previous decoder
                dy = self.din,
                dk = self.dh,
                dv = self.dh,
                mask = self.mask,
                act = self.act
            )
        self.mh_self_attention_norm_layer = nn.LayerNorm(self.h*self.dh)
        self.mh_cross_attention_layer = MultiHeadAttention(
                h = self.h,
                dx = self.dx,
                dy = self.h*self.dh, # y will be the output of the mh_self_attention_norm_layer 
                dk = self.dh,
                dv = self.dh,
                mask = None,    # Paper does not indicate the presence of masking in this layer
                act = self.act
            )
        self.mh_cross_attention_norm_layer = nn.LayerNorm(self.h*self.dh)
        self.feed_forward_layer = nn.Sequential(
                nn.Linear(self.h*self.dh, self.dout),
                GELU(),
            )
        self.feed_forward_norm_layer = nn.LayerNorm(self.dout)

    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """ Function to calculate the forward pass in a decoder block.

        Inputs:
            - x:    sequence of encoded/embedded tokens. Relates to the
                    variable x used in the attention layer -> will be used
                    to calculate the Query tensor.
            - y:    Sequence of of "partially decoded tokens" passed on by
                    previous decoder blocks. Relates to the variable y used
                    in the attention layer -> will be used to calulate the
                    Key and Value tensor.
        """
        result_self_attention = self.mh_self_attention_layer.forward(y, y)
        result_self_attention_norm = torch.Tensor
        if self.din == self.h*self.dh:
            result_self_attention_norm = self.mh_self_attention_norm_layer(y+result_self_attention)
        else: 
            #Skipping the residual path
            print("Skipping residual path in self attention in DecoderBlock")
            result_self_attention_norm = self.mh_self_attention_norm_layer(result_self_attention)
        result_cross_attention = self.mh_cross_attention_layer.forward(x, result_self_attention_norm)
        result_cross_attention_norm = self.mh_cross_attention_norm_layer(result_cross_attention + result_self_attention_norm)
        result_feed_forward = self.feed_forward_layer(result_cross_attention_norm)
        result_feed_forward_norm = torch.Tensor
        if self.h*self.dh == self.dout:
            result_feed_forward_norm = self.feed_forward_norm_layer(result_feed_forward + result_cross_attention_norm)
        else:
            # Skipping residual path
            print("Skipping residual path after feed forward layer in DecoderBlock")
            result_feed_forward_norm = self.feed_forward_norm_layer(result_feed_forward)

        return result_feed_forward_norm

            
