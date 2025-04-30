import torch.nn as nn
import torch
input_tensor=torch.randn((2,3,5,8))
norm=nn.BatchNorm2d(3)
output_tensor=norm(input_tensor)
print(output_tensor)