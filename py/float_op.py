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

SEED = 1233

IN_C = 4
OUT_C = 16
NUM_SAMPLES=256
INPUT_H=2
INPUT_W=2
PTQ_BIT=8
assert KERNEL_SIZE == 1, "only kernel == 1 implemented"
torch.no_grad()
random.seed(SEED)
np.random.seed(SEED)

if __name__ == "__main__":
    np_input_float = np.random.random((1, IN_C, INPUT_H, INPUT_W))
    conv_bias = np.random.random((OUT_C))
    conv_weight = np.random.random((OUT_C, IN_C, KERNEL_SIZE, KERNEL_SIZE))

    print("create conv weight, shape:{}, dtype:{}".format(conv_weight.shape, conv_weight.dtype))
    print("create conv bias, shape:{}, dtype:{}".format(conv_bias.shape, conv_bias.dtype))

    # create torch conv module with specified weight and bias
    torch_conv_float = create_conv_1x1(conv_weight, conv_bias, in_c=IN_C, out_c=OUT_C)

    # calculate using torch
    torch_output_float = torch_conv_float(torch.from_numpy(np_input_float)).detach().numpy()

    # calculate using hand write numpy ops
    np_output_float = do_np_conv_1x1(conv_weight, conv_bias, np_input_float)

    # compare hand write float op with torch
    print("Testing if hand write float conv equals to torch")
    np.testing.assert_allclose(torch_output_float, np_output_float)
    print("\033[032m[ Test passed !!!]\033[0m")

    # prepare ptq samples
    ptq_inputs_float = np.random.random((NUM_SAMPLES, IN_C, INPUT_H, INPUT_W))

    # do ptq
    ptq_conv = ptq(conv_weight, conv_bias, ptq_inputs_float, PTQ_BIT)
    
    # do quant conv
    ptq_output_float = ptq_conv(np_input_float)
    
    # compare result
    print("oritin result:{}".format(torch_output_float.flatten()))
    print("diff:{}".format(torch_output_float.flatten() - ptq_output_float.flatten()))
