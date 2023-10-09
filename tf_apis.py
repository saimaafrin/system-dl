import tensorflow as tf
import gp_apis

def Mul(input1, input2, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return Mul_real(X1, X2, dim;_0, dim;_1, device0)
    return _lambda(input1, input2)

def Mul_real(input1, input2, dim;_0, dim;_1, device0):
    out = gp_apis.gp_Mul(input1, input2, dim;_0, dim;_1, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_Mul(dZ1, dZ2, dim;_0, dim;_1, device0)
    return out, grad

def Tpose(input1, dim;_0, dim;_1, device0):
    @tf.custom_gradient
    def _lambda(X1):
        return Tpose_real(X1, dim;_0, dim;_1, device0)
    return _lambda(input1)

def Tpose_real(input1, dim;_0, dim;_1, device0):
    out = gp_apis.gp_Tpose(input1, dim;_0, dim;_1, device0)
    def grad(dZ1):
        return gp_apis.gp_Tpose(dZ1, dim;_0, dim;_1, device0)
    return out, grad

