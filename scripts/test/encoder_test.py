from model.encoder import Encoder, EncoderStack
import torch


x = torch.Tensor([[1, 0, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2]])
y = torch.Tensor([[.8, .2], [.8, .2], [1, 0], [0, 1]])


# Testing Encoder Layer
print("\n#------ Testing encoder layer ------")
model = Encoder(h=3, dh=4, dx=3, d_layer=12)
result = model.forward(x)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


# Testing Encoder Stack 
print("\n#------ Testing encoder stack ------")
model = EncoderStack(n=3, h=3, dh=4, dx=3, d_layer=12)
result = model.forward(x)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
