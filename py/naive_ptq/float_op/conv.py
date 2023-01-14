import torch
import numpy as np
KERNEL_SIZE = 1

def create_conv_1x1(weight, bias, in_c, out_c):
    assert weight.shape == (out_c, in_c, 1, 1)
    assert bias.shape == (out_c,)
    torch_conv_float = torch.nn.Conv2d(in_c, out_c, KERNEL_SIZE, 1, 0)
    torch_conv_float.weight.data = torch.from_numpy(weight)
    torch_conv_float.bias.data = torch.from_numpy(bias)
    return torch_conv_float

def do_np_conv_1x1(weight, bias, input):
    """Do simplist 1x1 conv with out dilation and padding

    Args:
        weight(np.array): conv weight, shape: (out_c, in_c, kernel_size, kernel_size)
        bias(np.array): conv bias, shape: (out_c, )
        input(np.array): input, shape: (n, c, h, w)
    """
    out_c = weight.shape[0]
    in_c = weight.shape[1]
    input_h = input.shape[2]
    input_w = input.shape[3]

    assert weight.shape[2] == KERNEL_SIZE
    assert weight.shape[3] == KERNEL_SIZE
    assert out_c == bias.shape[0]
    assert in_c == input.shape[1]
    reshape_input = input.reshape((in_c, -1))
    multi_result = np.matmul(weight.reshape((out_c, in_c)), reshape_input)
    return (multi_result + np.tile(bias.reshape(-1, 1), (1, input_h * input_w))).reshape((1, out_c, input_h, input_w))