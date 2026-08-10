from attention import Attention
from encoder import Encoder
import torch
from torch.nn.modules.activation import ReLU


x = torch.Tensor([[1, 0, 2], [0, 1, 2]])
y = torch.Tensor([[.8, .2], [.8, .2], [.5, .5]])


# Testing Attention Layer
model = Attention(dx=3, dy=2, dk=4, dv=4, act=ReLU)

result = model.forward(x, y)
print(result)


# Testing Encoder Layer
model = Encoder(dx=3, d_model=4)
result = model.forward(x)
print(result)

