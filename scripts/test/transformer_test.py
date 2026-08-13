import torch
from model.transformer import Transformer


x = torch.Tensor([[1, 0, 1], [0, 1, 2], [1, 0, 3], [0, 1, 4]])
y = torch.Tensor([[.8, .2], [.8, .2], [1, 0], [0, 1]])


# Testing Decoder Stack
print("\n#------ Testing Transformer ------")
model = Transformer(
        dx = 3,
        ne = 3,
        nd = 3, 
        nh = 3, 
        dh = 3,
    )
result = model.forward(x)
print(result)
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

