# copyright@ Minghao Gou 2023
# email: gouminghao@gmail.com

import numpy as np
import math

from .quantization import (
    SUPPORTED_BITS,
    SUPPORTED_WEIGHT_BITS,
    BIAS_MAP,
    QuantizedArray,
    QuantizedConv
)

def minmax_quantization(inputs, num_bits=8):
    """
    Quantize array with minmax method.

    Args:
        inputs(np.array): samples.
        num_bits(int): how many bits to quantize

    Returns:
        int: quanization shiftbit
    """
    assert num_bits in SUPPORTED_BITS
    max_val = np.max(np.abs(inputs))
    shiftbit = num_bits - 1 - int(math.log(max_val, 2)) # 1 for sign bit
    return shiftbit

def ptq(weight, bias, inputs, num_bit):
    """Do ptq(post training quantization) on a simple conv.


    Args:
        weight(np.array): float conv weight
        bias(np.array): float conv bias
        inputs(np.array): float inputs
        num_bit(int): quantization bits.
    
    Returns:
        QuantizedConv: quantized module.
    """
    assert num_bit in SUPPORTED_WEIGHT_BITS, "quantized num bit must be in {}".format(SUPPORTED_WEIGHT_BITS)
    weight_shift_bit = minmax_quantization(weight)
    input_shift_bit = minmax_quantization(inputs)
    bias_shift_bit = weight_shift_bit + input_shift_bit
    q_weight = QuantizedArray(weight, weight_shift_bit, num_bit)
    q_bias = QuantizedArray(bias, bias_shift_bit, BIAS_MAP[num_bit])
    return QuantizedConv(q_weight, q_bias, input_shift_bit, num_bit)
