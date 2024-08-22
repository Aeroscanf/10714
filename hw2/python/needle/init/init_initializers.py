import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Calculate the scale factor 'a'
    a = gain * (6 / (fan_in + fan_out)) ** 0.5
    
    # Generate random values between 0 and 1
    values = rand(fan_in, fan_out)
    
    # Scale and shift to get values between -a and a
    values = 2 * a * values - a
    
    # Create and return an ndl.Tensor with these values
    return ndl.Tensor(values, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    #caculate the standard divation
    std = gain * ((2.0 / (fan_in + fan_out)) ** 0.5)
    
    z = std * randn(fan_in, fan_out)
    return ndl.Tensor(z, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = 2.0 ** 0.5
    bound = gain * ( (3.0 / fan_in) ** (0.5))

    values = rand(fan_in, fan_out)

    values = 2 * bound * values - bound

    return ndl.Tensor(values, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = 2.0 ** 0.5
    std = gain / ((fan_in) ** 0.5)

    z = std * randn(fan_in, fan_out)

    return ndl.Tensor(z, **kwargs)
    ### END YOUR SOLUTION
