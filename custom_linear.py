import torch
import torch.nn as nn
from torch.autograd import Function
from pytorch_apis import Mul, Tpose
from gp_apis import gp_Mul, gp_Tpose
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


'''
A = torch.rand(dim_m, dim_k, dtype=torch.float32)
B = torch.rand(dim_k, dim_n, dtype=torch.float32)
C_gpu = Mul(A, B, dim_m, dim_n, dim_m,  dim_k, dim_n, device)  # int numARows, int numAColumns, int numBColumns
Ct_gpu = Tpose(A, dim_k, dim_m, dim_m, dim_k, device) #Arows, Acols '''

class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, device):

        print("Input size[0]:", input.size(0))   
        print("Input size[1]:", input.size(1))    
        print("******* INPUT X ",input)
        print("Weight size[0]:", weight.size(0))
        print("Weight size[1]:", weight.size(1)) 
        print("Weight Tensor", weight)    
     
        print("Bias size:", bias.size())
        print("Bias Tensor", bias)

        ctx.save_for_backward(input, weight, bias)
        ctx.device = device
    
        output = gp_Mul(input, weight, input.size(0), weight.size(0), input.size(0), input.size(1), weight.size(1), device)
        
        print("Output size before adding bias:", output.size()) 
        print("Output before adding bias", output)
        output += bias
        print("Output after adding bias", output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        device = ctx.device

        print("---------------------------- Grad output",grad_output)
        
       # Compute gradient with respect to input
        print("Weight size 0, 1",weight.size(0), weight.size(1))
        print("grad_output.size 0, 1",grad_output.size(0), grad_output.size(1))
        transposed_weight = gp_Tpose(weight, weight.size(1), weight.size(0), weight.size(0), weight.size(1), device)
        print("Trans_W size 0, 1",transposed_weight.size(0), transposed_weight.size(1))
        grad_input = gp_Mul(grad_output, transposed_weight, grad_output.size(0),transposed_weight.size(0), grad_output.size(0), grad_output.size(1), transposed_weight.size(1), device)
        print("grad input dX 0, 1", grad_input.size(0), grad_input.size(1))
    
        print("********************************************************************")
        # Compute gradient with respect to weight
       
        print("Input size 0, 1",input.size(0), input.size(1))
        print("grad_output.size 0, 1",grad_output.size(0), grad_output.size(1))
        transposed_input = gp_Tpose(input, input.size(1), input.size(0), input.size(0), input.size(1), device)
        print("Trans_Input size 0, 1",transposed_input.size(0), transposed_input.size(1))
        grad_weight = gp_Mul(transposed_input, grad_output, grad_output.size(1), transposed_input.size(0), transposed_input.size(0), transposed_input.size(1), grad_output.size(1), device)
        print("grad weight dW 0, 1", grad_weight.size(0), grad_weight.size(1))
        
       # Compute gradient with respect to bias
        grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias, None

class CustomLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(CustomLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device))
        self.bias = nn.Parameter(torch.Tensor(out_features).to(device))
        self.device = device

        # Initialize weights and biases
        nn.init.xavier_uniform_(self.weight)
        nn.init.uniform_(self.bias)


    def forward(self, input):
        return CustomLinearFunction.apply(input, self.weight, self.bias, self.device)

