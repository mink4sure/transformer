from attention import Attention
import torch

x = torch.Tensor([[1, 0, 2], [0, 1, 2]])
y = torch.Tensor([[.8, .2], [.8, .2], [.5, .5]])

model = Attention(dx=3, dy=2, dk=4, dv=4)

result = model.forward(x, y)
print(result)
