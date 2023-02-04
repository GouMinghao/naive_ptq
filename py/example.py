# copyright@ Minghao Gou 2023
# email: gouminghao@gmail.com

import torch
import numpy as np
import cv2
import os
import json

import random

from naive_ptq.ptq.ptq import ptq
from naive_ptq.float_op.conv import KERNEL_SIZE, create_conv_1x1, do_np_conv_1x1

SEED = 1233  # random seed to make result reproducable

IN_C = 4
OUT_C = 8
NUM_SAMPLES = 256  # ptq samples
INPUT_H = 20
INPUT_W = 30
PTQ_BIT = 8
DUMP_DIR = "dumps"
os.makedirs(DUMP_DIR, exist_ok=True)
assert KERNEL_SIZE == 1, "only kernel == 1 implemented"

torch.no_grad()
random.seed(SEED)
np.random.seed(SEED)

def show_dump(dump_dir):
    input_32 = np.fromfile(os.path.join(dump_dir, "q_input_asint32.bin"), dtype=np.int32)
    output_32 = np.fromfile(os.path.join(dump_dir, "q_output_asint32.bin"), dtype=np.int32)
    print("input 32: {}".format(input_32))
    print("output_32: {}".format(output_32))

if __name__ == "__main__":
    # create input and conv parameters
    np_input_float = np.random.random((1, IN_C, INPUT_H, INPUT_W))
    conv_bias = np.random.random((OUT_C))
    conv_weight = (np.random.random((OUT_C, IN_C, KERNEL_SIZE, KERNEL_SIZE)) - 0.5) * 4

    print(
        "create conv weight, shape:{}, dtype:{}".format(
            conv_weight.shape, conv_weight.dtype
        )
    )
    print(
        "create conv bias, shape:{}, dtype:{}".format(conv_bias.shape, conv_bias.dtype)
    )
    # create torch conv module with specified weight and bias
    torch_conv_float = create_conv_1x1(conv_weight, conv_bias, in_c=IN_C, out_c=OUT_C)

    # calculate using torch
    torch_output_float = (
        torch_conv_float(torch.from_numpy(np_input_float)).detach().numpy()
    )

    # calculate using hand write numpy ops
    np_output_float = do_np_conv_1x1(conv_weight, conv_bias, np_input_float)

    # compare hand write float op with torch
    print("Testing if hand write float conv equals to torch")
    np.testing.assert_allclose(torch_output_float, np_output_float)
    print("\033[032m[ Test passed !!!]\033[0m")

    # prepare ptq samples
    ptq_inputs_float = np.random.random((NUM_SAMPLES, IN_C, INPUT_H, INPUT_W))

    # do ptq
    ptq_conv = ptq(conv_weight, conv_bias, ptq_inputs_float, PTQ_BIT, DUMP_DIR)

    # do quant conv
    ptq_output_float = ptq_conv(np_input_float)

    # compare result
    print("torch result:{}".format(torch_output_float.flatten()))
    print("quant result:{}".format(ptq_output_float.flatten()))
    diff = np.abs(torch_output_float.flatten() - ptq_output_float.flatten())
    print("diff:{}".format(diff))
    print("relative diff:{}".format(diff / np.abs(torch_output_float.flatten())))
    show_dump(DUMP_DIR)