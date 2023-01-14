# copyright@ Minghao Gou 2023
# email: gouminghao@gmail.com

import numpy as np
import json
import os

SUPPORTED_BITS = [8, 16, 32, 64]
SUPPORTED_WEIGHT_BITS = [8, 16]

BITS_MAP = {
    8: np.int32,
    16: np.int64,
    32: np.int64,
    64: np.int64,
}  # map from weight bit to np arr type

BIAS_MAP = {8: 32, 16: 64}  # map from weight bit to bias bit


class QuantizedArray:
    def __init__(self, arr, shiftbit, num_bit, do_quantize=True):
        """
        Quantized array inplementation

        Args:
            arr(np.array): the float point array to be quantized or int array.
            shiftbit(int): shift bits.
            num_bit(int): number of bits to quantize.
        """
        assert num_bit in SUPPORTED_BITS, "quantized num bit must be in {}".format(
            SUPPORTED_BITS
        )
        self.num_bit = num_bit
        self.shiftbit = shiftbit
        self.scale = 2**self.shiftbit
        self.max_scale = 2**self.num_bit
        if do_quantize:
            self.int_arr = np.round(arr * self.scale).astype(
                BITS_MAP[self.num_bit]
            )  # do scale and round
            self.int_arr = np.clip(self.int_arr, -self.max_scale, self.max_scale - 1)
        else:
            assert arr.dtype in [np.int32, np.int64]
            self.int_arr = arr

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
        Do fixed point matrix multiplication instead of elementsize.

        Args:
            other(QuantizedArray): the other operator.
        """
        assert self.num_bit == other.num_bit
        return QuantizedArray(
            np.matmul(self.int_arr, other.int_arr),
            self.shiftbit + other.shiftbit,
            self.num_bit,
            do_quantize=False,
        )

    def __add__(self, other):
        """Do fixed point addition.

        Args:
            other(QuantizedArray): the other operator.
        """
        assert self.shiftbit == other.shiftbit
        # assert self.num_bit == other.num_bit
        return QuantizedArray(
            self.int_arr + other.int_arr, self.shiftbit, self.num_bit, do_quantize=False
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
        save_dir=None,
    ):
        self.weight = weight
        self.bias = bias
        self.input_shift_bit = input_shift_bit
        self.save_dir = save_dir
        if self.save_dir is not None:
            self.export_json(os.path.join(self.save_dir, "conv.json"))

    def export_json(self, json_path):
        ckpt = {
            "weight": {
                "num_bit": self.weight.num_bit,
                "shift_bit": self.weight.shiftbit,
                "arr": self.weight.int_arr.tolist(),
            },
            "bias": {
                "num_bit": self.bias.num_bit,
                "shift_bit": self.bias.shiftbit,
                "arr": self.bias.int_arr.tolist(),
            },
            "input_shiftbit": self.input_shift_bit,
        }
        with open(json_path, "w") as json_f:
            json.dump(ckpt, json_f)

    def load_json(self, json_path):
        with open(json_path, "r") as json_f:
            ckpt = json.load(json_f)
        self.input_shift_bit = ckpt["input_shiftbit"]
        weight_num_bit = ckpt["weight"]["num_bit"]
        weight_shiftbit = ckpt["weight"]["shift_bit"]
        weight_arr = np.array(ckpt["weight"]["arr"], dtype=BITS_MAP[weight_num_bit])
        bias_num_bit = ckpt["bias"]["num_bit"]
        bias_shiftbit = ckpt["bias"]["shift_bit"]
        bias_arr = np.array(ckpt["bias"]["arr"], dtype=BITS_MAP[bias_num_bit])
        self.weight = QuantizedArray(
            weight_arr, weight_shiftbit, weight_num_bit, do_quantize=False
        )
        self.bias = QuantizedArray(
            bias_arr, bias_shiftbit, bias_num_bit, do_quantize=False
        )

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
        quantized_input = QuantizedArray(
            reshape_input, self.input_shift_bit, self.weight.num_bit
        )
        if self.save_dir is not None:
            info_dict = {}
            info_dict["input_shiftbit"] = quantized_input.shiftbit
            info_dict["input_shape"] = list(input.shape)
            info_dict["input_numbit"] = quantized_input.num_bit
            quantized_input.int_arr.tofile(
                os.path.join(
                    self.save_dir, "q_input_int{}.bin".format(quantized_input.num_bit)
                )
            )
        # do fixed point conv
        quantized_output = self.weight.reshape((out_c, in_c)) * quantized_input
        tile_bias = self.bias.reshape((-1, 1)).tile((1, input_h * input_w))
        quantized_output += tile_bias
        quantized_output = quantized_output.reshape((1, out_c, input_h, input_w))
        if self.save_dir is not None:
            info_dict["output_shiftbit"] = quantized_output.shiftbit
            info_dict["output_shape"] = list(quantized_output.int_arr.shape)
            info_dict["output_numbit"] = quantized_output.num_bit
            quantized_output.int_arr.tofile(
                os.path.join(
                    self.save_dir, "q_output_int{}.bin".format(quantized_output.num_bit)
                )
            )
            with open(
                os.path.join(self.save_dir, "quant_conv_info.json"), "w"
            ) as json_f:
                json.dump(info_dict, json_f)
        # dequantize and output
        return quantized_output.Dequantize()
