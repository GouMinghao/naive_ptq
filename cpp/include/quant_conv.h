#pragma once

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

#include "utils.h"

Eigen::MatrixXi quantize(Eigen::MatrixXd float_array, int32_t shiftbit);
Eigen::MatrixXi dequantize(Eigen::MatrixXd quant_array, int32_t shiftbit);

class QuantConv
{
 public:
    void show();
    QuantConv();
    QuantConv(std::string json_config);
    Eigen::MatrixXi call(
      Eigen::MatrixXi input_i,
      uint32_t c,
      uint32_t h,
      uint32_t w);
    Eigen::MatrixXd call(Eigen::MatrixXd input_d);

 private:
    int32_t weight_shiftbit;
    int32_t weight_num_bit;
    int32_t bias_shiftbit;
    int32_t bias_num_bit;
    int32_t input_shiftbit;
    Eigen::MatrixXi weight;
    Eigen::MatrixXi bias;
};
