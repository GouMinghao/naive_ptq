#pragma once

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

class Tensor {
 private:
    uint32_t _c;
    uint32_t _h;
    uint32_t _w;
    Eigen::MatrixXi _val;
 public:
    Tensor();
    Tensor(uint32_t c, uint32_t h, uint32_t w, Eigen::MatrixXi val);
    Eigen::MatrixXi to_matrix(uint32_t h, uint32_t w);
};