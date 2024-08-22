from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        #Z_data = Z.realize_cached_data() if isinstance(Z, Tensor) else Z
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_Z = array_api.exp(Z - max_Z)
        sum_exp_Z = array_api.sum(exp_Z, axis=self.axes, keepdims = True)
        result = array_api.log(sum_exp_Z) + max_Z
        if self.axes is not None:
            result = array_api.squeeze(result, axis = self.axes)
        else:
            result = array_api.squeeze(result)
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        Z = node.inputs[0]
        Z_data = Z.realize_cached_data()
        max_Z = array_api.max(Z_data, axis=self.axes, keepdims=True)
    
        # Convert numpy arrays to Tensors before operations
        Z_centered = Tensor(Z_data - max_Z)
        exp_Z = exp(Z_centered)
        sum_exp_Z = summation(exp_Z, axes=self.axes)
    
        grad_sum_exp_Z = out_grad / sum_exp_Z
        expand_shape = list(Z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_Z = grad_sum_exp_Z.reshape(expand_shape).broadcast_to(Z.shape)
        return grad_exp_Z * exp_Z



def logsumexp(a, axes = None):
    return LogSumExp(axes = axes)(a)

