import numpy as np
import torch
from torch import nn


class Attention(nn.Module):
    """ Generic attention layer 

    Attention(Q,V,K) = matmul(softmax(matmul(Q,K.T)/sqrt(dk)),V)

    Parameters:
        - dx:   Dimensionality of the tensor from which to calulate the Queries
        - dy:   Dimensionality of the tensor form which to calculate the Keys and Values
        - dk:   Dimensionality of the Queries and Keys
        - dv:   Dimensionality of the Values
        - mask: Function to mask the matmul(Q,K.T) tensor
    """        
    def __init__(self, dx: int, dy: int, dk: int, dv: int, mask=None):
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.dk = dk
        self.dv = dv
        self.mask = mask

        self.query_layer = nn.Linear(dx, dk, bias=False)
        self.key_layer = nn.Linear(dy, dk, bias=False)
        self.value_layer = nn.Linear(dy, dv, bias=False)
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
            attention = self.mask(attention)

        attention = self.softmax_layer(attention)
        attention = torch.matmul(attention, v)
        
        return attention


    def forward(self, x, y):
        return self.attention(x, y)
    

class MultiHeadAttention(nn.Module):
    """ Simple multi head attention class

    I'm going to make the computationally ineffient decision and just run
    serveral attention layers seperately and concat their results. Other 
    implementations smartly utilize reshape and transpose functions to
    limit the ammount of matrix operations.

    Parameters:
        - h:    Number of heads
        - dx:   Dimensionality of the tokens/embeddings from which to calculate
                Query tensor.
        - dy:   Dimensionality of the tokens/embeddings from which to calculate
                the Key and Value tensors.
        - dk:   Dimensionality of the Query and Key tensors in each head
        - dv:   Dimensionality of the Value tensor in each head
        - mask:
    """
    def __init__(self, h, dx, dy, dk, dv, mask=None):
        super().__init__()
        self.h = h
        self.dx = dx
        self.dy = dy
        self.dk = dk
        self.dv = dv
        self.mask = mask

        self.attention_heads = nn.ModuleList(
                [Attention(self.dx, self.dy, self.dk, self.dv, self.mask) for i in range(self.h)]
            )
        
    def forward(self, x, y):
        concatted_attention = torch.Tensor()
        for i, a in enumerate(self.attention_heads):
            attention_head_i = a.forward(x, y)
            concatted_attention = torch.cat((concatted_attention, attention_head_i), dim=-1)
        return concatted_attention
