import torch as th
import gp_apis

class Mul_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, device0):
        res = gp_apis.gp_Mul(input1, input2, dim_0, dim_1, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def Mul(input1, input2, dim_0, dim_1, device0):
    return Mul_impl.apply(input1, input2, dim_0, dim_1, device0)

class Tpose_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, dim_0, dim_1, device0):
        res = gp_apis.gp_Tpose(input1, dim_0, dim_1, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def Tpose(input1, dim_0, dim_1, device0):
    return Tpose_impl.apply(input1, dim_0, dim_1, device0)

