import torch
import gp_apis
#import custom_linear
from custom_linear import CustomLinearLayer 


device = torch.device("cuda:0")
a = torch.ones(4,5).to(device)
b = torch.ones(5,6).to(device)

c_torch = torch.mm(a,b)
c_gp    = gp_apis.gp_Mul(a, b, 4, 6, device)

print(c_torch)
print(c_gp)

