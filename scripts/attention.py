import numpy as np
import torch
from torch import nn
from torch.nn.modules.activation import GELU


class Attention(nn.Module):
    """ Generic attention block

    Attention(Q,V,K) = matmul(softmax(matmul(Q,K.T)/sqrt(dk)),V)

    """        
    def __init__(self, dx: int, dy: int, dk: int, dv: int, mask=None, act=GELU):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dk = dk
        self.dv = dv
        self.mask = mask

        self.query_layer = nn.Sequential(
                nn.Linear(dx, dk, bias=False),
                act(),
            )
        self.key_layer = nn.Sequential(
                nn.Linear(dy, dk, bias=False),
                act(),
            )
        self.value_layer = nn.Sequential(
                nn.Linear(dy, dv, bias=False),
                act(),
            )
        self.softmax_layer = nn.modules.activation.Softmax(dim=-1)


    def attention(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        A function to calculate the attention between two tensors x and y.

        Inputs:
         - x (torch.Tensor):    sequence of tokens/embeddings from which 
                                to calculate the Query tensors. 
         - y (torch.Tensor):    sequence of tokens/embeddings from which
                                to calculate the Key and Value tensors.

        Returns:
         - Attention(Q,K,V):    Not yet sure what form this will have
        """
        q = self.query_layer(x)
        k = self.key_layer(y)
        v = self.value_layer(y)
        
        kt = torch.transpose(k, 0, 1)
        attention = torch.matmul(q, kt)/np.sqrt(self.dk)
        
        if self.mask is not None:
            assert False, "Masking in attention not yet implemented"

        attention = self.softmax_layer(attention)
        attention = torch.matmul(attention, v)
        
        return attention


    def forward(self, x, y):
        return self.attention(x, y)
