import torch
from model.decoder import DecoderBlock, DecoderStack


x = torch.Tensor([[1, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2]])
y = torch.Tensor([[.8, .2], [.8, .2], [1, 0], [0, 1]])


# Testing Decoder Block
print("\n#------ Testing Decoder Block ------")
model = DecoderBlock(din=2, h=3, dh=4, dx=3, dout=12)
result = model.forward(x, y)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Testing Decoder Stack
print("\n#------ Testing Decoder Stack ------")
model = DecoderStack(n=4, h=3, dh=4, dx=3, dout=12)
result = model.forward(x)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

