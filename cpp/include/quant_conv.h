#pragma once

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

Eigen::MatrixXi quantize(Eigen::MatrixXd float_array, int32_t shiftbit);
Eigen::MatrixXi dequantize(Eigen::MatrixXd quant_array);

class QuantConv
{
 public:
    QuantConv();
    QuantConv(std::string json_config);
    Eigen::MatrixXi call(Eigen::MatrixXi input);
 private:
    Eigen::MatrixXi kernel;
    Eigen::MatrixXi bias;
};
