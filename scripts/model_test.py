from attention import Attention, MultiHeadAttention
from encoder import Encoder, EncoderStack
import torch
from torch.nn.modules.activation import ReLU


x = torch.Tensor([[1, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2]])
y = torch.Tensor([[.8, .2], [.8, .2], [1, 0], [0, 1]])


# Testing Attention Layer
print("#------ Testing attention layer ------")
model = Attention(dx=3, dy=2, dk=4, dv=4, act=ReLU)
result = model.forward(x, y)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Testing MultiHeadAttention Layer
print("#------ Testing multi head attention layer ------")
model = MultiHeadAttention(h=3, dx=3, dy=2, dk=2, dv=2, act=ReLU)
result = model.forward(x, y)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Testing Encoder Layer
print("\n#------ Testing encoder layer ------")
model = Encoder(dx=3, d_model=4)
result = model.forward(x)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Testing Encoder Layer
print("\n#------ Testing encoder stack ------")
model = EncoderStack(n=3, dx=3, d_model=4)
result = model.forward(x)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
