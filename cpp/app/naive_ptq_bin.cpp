#include "quant_conv.h"
#include <iostream>

int main()
{
    QuantConv quant_conv(std::string("/home/gmh/git/naive_ptq/py/dumps/conv.json"));
    quant_conv.show();
    return 0;
}