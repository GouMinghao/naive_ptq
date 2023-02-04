#include <iostream>
#include "quant_conv.h"
#include "tensor.h"

int main()
{
    QuantConv quant_conv(std::string("/home/gmh/git/naive_ptq/py/dumps/conv.json"));
    Tensor<int32_t> q_int32_in(4, 2, 2, "/home/gmh/git/naive_ptq/py/dumps/q_input_asint32.bin");
    Tensor<int32_t> q_int32_out_gt(8, 2, 2, "/home/gmh/git/naive_ptq/py/dumps/q_output_asint32.bin");
    Tensor<int32_t> q_int32_out = quant_conv.call(q_int32_in);
    q_int32_in.show();
    q_int32_out.show();
    q_int32_out_gt.show();
    return 0;
}