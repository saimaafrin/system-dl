import torch as th
import gp_apis

class gspmmv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, dim_0, dim_1, reverse, norm, device0):
        res = gp_apis.gp_gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmv(graph, input1, dim_0, dim_1, reverse, norm, device0):
    return gspmmv_impl.apply(graph, input1, dim_0, dim_1, reverse, norm, device0)

class gspmmve_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmve(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0):
    return gspmmve_impl.apply(graph, input1, edge_input, dim_0, dim_1, op, reverse, device0)

class gspmme_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, edge_input, dim_0, op, reverse, device0):
        res = gp_apis.gp_gspmme(graph, edge_input, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmme(graph, edge_input, dim_0, op, reverse, device0):
    return gspmme_impl.apply(graph, edge_input, dim_0, op, reverse, device0)

class gspmme2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, edge_input, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmme2d(graph, edge_input, dim_0, dim_1, op, reverse, device0):
    return gspmme2d_impl.apply(graph, edge_input, dim_0, dim_1, op, reverse, device0)

class gspmmve2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
        res = gp_apis.gp_gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gspmmve2d(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0):
    return gspmmve2d_impl.apply(graph, input1, edge_input, dim_0, dim_1, dim_2, op, reverse, device0)

class gsddmmve_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, op, reverse, device0):
        res = gp_apis.gp_gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmve(graph, input_left, input_right, dim_0, op, reverse, device0):
    return gsddmmve_impl.apply(graph, input_left, input_right, dim_0, op, reverse, device0)

class gsddmmve2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmve2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    return gsddmmve2d_impl.apply(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)

class gsddmmvv_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, op, reverse, device0):
        res = gp_apis.gp_gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmvv(graph, input_left, input_right, dim_0, op, reverse, device0):
    return gsddmmvv_impl.apply(graph, input_left, input_right, dim_0, op, reverse, device0)

class gsddmmvv2d_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
        res = gp_apis.gp_gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def gsddmmvv2d(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0):
    return gsddmmvv2d_impl.apply(graph, input_left, input_right, dim_0, dim_1, op, reverse, device0)

class test_2out_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
        res1, res2 = gp_apis.gp_test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res1, res2

    @staticmethod
    def backward(ctx, dZ1, dZ2):
        pass #must be implemented

def test_2out(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    return test_2out_impl.apply(graph, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)

class test3_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
        res1, res2 = gp_apis.gp_test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)
        ctx.backward_cache = None #must be implemented
        return res1, res2

    @staticmethod
    def backward(ctx, dZ1, dZ2):
        pass #must be implemented

def test3(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0):
    return test3_impl.apply(input1, input2, dim1_0, dim1_1, dim2_0, dim2_1, op, reverse, device0)

class test4_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
        res = gp_apis.gp_test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def test4(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0):
    return test4_impl.apply(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, t, device0)

class vectorAdd_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, W, device0):
        res = gp_apis.gp_vectorAdd(input1, input2, dim_0, W, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def vectorAdd(input1, input2, dim_0, W, device0):
    return vectorAdd_impl.apply(input1, input2, dim_0, W, device0)

class Mul_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, dim_0, dim_1, numARows, numAColumns, numBColumns, device0):
        res = gp_apis.gp_Mul(input1, input2, dim_0, dim_1, numARows, numAColumns, numBColumns, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def Mul(input1, input2, dim_0, dim_1, numARows, numAColumns, numBColumns, device0):
    return Mul_impl.apply(input1, input2, dim_0, dim_1, numARows, numAColumns, numBColumns, device0)

class Tpose_impl(th.autograd.Function):
    @staticmethod
    def forward(ctx, input1, dim_0, dim_1, rows, cols, device0):
        res = gp_apis.gp_Tpose(input1, dim_0, dim_1, rows, cols, device0)
        ctx.backward_cache = None #must be implemented
        return res

    @staticmethod
    def backward(ctx, dZ):
        pass #must be implemented

def Tpose(input1, dim_0, dim_1, rows, cols, device0):
    return Tpose_impl.apply(input1, dim_0, dim_1, rows, cols, device0)

