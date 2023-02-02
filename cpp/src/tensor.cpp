#include "tensor.h"

Tensor::Tensor(uint32_t c, uint32_t h, uint32_t w, Eigen::MatrixXi val)
{
    this -> _c = c;
    this -> _h = h;
    this -> _w = w;
    this -> _val = val;
}

Eigen::MatrixXi Tensor::to_matrix(uint32_t h, uint32_t w)
{
    
}