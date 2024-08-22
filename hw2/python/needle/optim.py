"""Optimization module"""
from types import DynamicClassAttribute
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

        for p in self.params:
            self.u[p] = ndl.zeros(*p.shape, dtype=p.dtype, device=p.device)

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.data

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            if self.momentum > 0:
                self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad
                update = self.u[p]
            else:
                update = grad

            update = ndl.Tensor(update, dtype=p.dtype)

            p.data = p.data - self.lr * update


    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

        for p in self.params:
            self.m[p] = ndl.zeros(*p.shape, dtype=p.dtype, device=p.device)
            self.v[p] = ndl.zeros(*p.shape, dtype=p.dtype, device=p.device)


    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data

            if self.weight_decay != 0:
                grad += self.weight_decay * p.data
            
            self.m[p] *= self.beta1
            self.m[p] += (1 - self.beta1)*grad
            self.v[p] *= self.beta2
            self.v[p] += (1-self.beta2)*(grad**2)

            u_hat = (self.m[p] / (1 - self.beta1 ** self.t)).realize_cached_data().astype(p.dtype)
            v_hat = (self.v[p] / (1 - self.beta2 ** self.t)).realize_cached_data().astype(p.dtype)

            p.data -= self.lr * u_hat / (v_hat**0.5 + self.eps)
            u_hat = None
            v_hat = None

        ### END YOUR SOLUTION
