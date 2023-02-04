#pragma once

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

#include "utils.h"
#include "tensor.h"

Eigen::MatrixXi quantize(Eigen::MatrixXd float_array, int32_t shiftbit);
Eigen::MatrixXi dequantize(Eigen::MatrixXd quant_array, int32_t shiftbit);

class QuantConv
{
 public:
    void show();
    void get_c();
    QuantConv();
    QuantConv(std::string json_config);
    Tensor<int32_t> call(Tensor<int32_t> input);
    Tensor<double> call(Tensor<double> input_d);
    int input_c;
    int output_c;
 private:
    int32_t weight_shiftbit;
    int32_t weight_num_bit;
    int32_t bias_shiftbit;
    int32_t bias_num_bit;
    int32_t input_shiftbit;
    Eigen::MatrixXi weight;
    Eigen::VectorXi bias;
};
