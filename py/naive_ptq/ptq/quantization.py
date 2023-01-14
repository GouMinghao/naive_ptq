# copyright@ Minghao Gou 2023
# email: gouminghao@gmail.com

import numpy as np

SUPPORTED_BITS = [8, 16, 32, 64]
SUPPORTED_WEIGHT_BITS = [8, 16]

BITS_MAP = {
    8: np.int32,
    16: np.int64,
    32: np.int64,
    64: np.int64
} # map from weight bit to np arr type

BIAS_MAP = {
    8: 32,
    16: 64
} # map from weight bit to bias bit

class QuantizedArray:
    def __init__(self, fp_arr, shiftbit, num_bit):
        """
        Quantized array inplementation

        Args:
            fp_arr(np.array): the float point array to be quantized.
            shiftbit(int): shift bits.
            num_bit(int): number of bits to quantize.
        """
        assert num_bit in SUPPORTED_BITS, "quantized num bit must be in {}".format(SUPPORTED_BITS)
        self.num_bit = num_bit
        self.shiftbit = shiftbit
        self.scale = 2 ** self.shiftbit
        self.int_arr = np.round(fp_arr * self.scale).astype(BITS_MAP[self.num_bit]) # do scale and round
        self.int_arr = np.clip(self.int_arr, - self.scale, self.scale - 1)

    @property
    def shape(self):
        return self.int_arr.shape

    def Dequantize(self):
        """
        convert to float number.

        Returns:
            np.array: dequantized float point array.
        """
        float_arr = self.int_arr.astype(np.float64)
        return float_arr / self.scale

    def __mul__(self, other):
        """
        Do fixed point multiplication.

        Args:
            other(QuantizedArray): the other operator.
        """
        assert self.num_bit == other.num_bit
        return QuantizedArray(
            np.matmul(self.int_arr, other.int_arr),
            self.shiftbit + other.shiftbit,
            self.num_bit
        )

    def __add__(self, other):
        """Do fixed point addition.

        Args:
            other(QuantizedArray): the other operator.
        """
        assert self.shiftbit == other.shiftbit
        # assert self.num_bit == other.num_bit
        return QuantizedArray(
            self.int_arr + other.int_arr,
            self.shiftbit,
            self.num_bit
        )

    def tile(self, shape):
        self.int_arr = np.tile(self.int_arr, shape)
        return self

    def reshape(self, shape):
        self.int_arr = self.int_arr.reshape(shape)
        return self

class QuantizedConv:
    def __init__(
        self,
        weight: QuantizedArray,
        bias: QuantizedArray,
        input_shift_bit,
        num_bit,
    ):
        self.weight = weight
        self.bias = bias
        self.input_shift_bit = input_shift_bit
        self.num_bit = num_bit
    
    def __call__(self, input):
        """Do quantized conv

        Args:
            input(np.array): float input

        Returns:

        """
        out_c = self.weight.shape[0]
        in_c = self.weight.shape[1]
        input_h = input.shape[2]
        input_w = input.shape[3]
        assert out_c == self.bias.shape[0]
        assert in_c == input.shape[1]

        # quantize input
        reshape_input = input.reshape((in_c, -1))
        quantized_input = QuantizedArray(reshape_input, self.input_shift_bit, self.num_bit)

        # do fixed point conv
        quantized_output = self.weight.reshape((out_c, in_c)) * quantized_input
        tile_bias = self.bias.reshape((-1, 1)).tile((1, input_h * input_w))
        quantized_output += tile_bias
        quantized_output = quantized_output.reshape((1, out_c, input_h, input_w))

        # dequantize and output
        return quantized_output.Dequantize()
