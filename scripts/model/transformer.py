import torch
from torch import nn
from torch.nn.modules.activation import GELU

from model.encoder import EncoderStack
from model.decoder import DecoderStack


class Transformer(nn.Module):
    """ The transformer model

    In this implementation the output of each encoder and decoder
    block will be determined by the number of heads times their
    dimensionality.

    The dimensionality of the output of the transformer will be
    the same as the input (dx)

    Parameters:
        - dx:   Dimensionality of the embedding
        - ne:   Number of encoders blocks
        - nd:   Number of decoders 
        - nh:   Number of heads
        - dh:   Dimensionality of each head
        """
    def __init__(self, dx, ne, nd, nh, dh, mask=torch.triu, act=GELU):
        super().__init__()
        self.dx = dx
        self.ne = ne
        self.nd = nd
        self.nh = nh
        self.dh = dh
        self.mask = mask
        self.act = act

        self.encoder_stack = EncoderStack(
                n = self.ne,
                dx = self.dx,
                h = self.nh,
                dh = self.dh,
                dout = self.nh*self.dh,
                act = self.act,
            )
        self.decoder_stack = DecoderStack(
                n = self.nd,
                dx = self.nh*self.dh,
                h = self.nh,
                dh = self.dh,
                dout = self.nh*self.dh,
                mask = self.mask,
                act = self.act,
            )
        self.linear = nn.Linear(self.nh*self.dh, dx)
        self.softmax = nn.modules.activation.Softmax(dim=-1)


    def forward(self, x):
        encoded = self.encoder_stack(x)
        decoded = self.decoder_stack(encoded)
        probs = self.softmax(self.linear(decoded))
        return probs

