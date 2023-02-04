#pragma once

#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

template<typename T>
class Tensor {
 public:
    Eigen::Matrix<T, -1, -1> _val;
    void show();
    uint32_t c;
    uint32_t h;
    uint32_t w;
    Tensor(uint32_t c, uint32_t h, uint32_t w, std::string file);
    Tensor(uint32_t c, uint32_t h, uint32_t w, Eigen::Matrix<T, -1, -1> val);
    Eigen::Matrix<T, -1, -1> to_matrix(uint32_t h, uint32_t w);
    bool operator==(Tensor other);
};

template<typename T>
Tensor<T>::Tensor(uint32_t c, uint32_t h, uint32_t w, Eigen::Matrix<T, -1, -1> val)
{
    this -> c = c;
    this -> h = h;
    this -> w = w;
    this -> _val = Eigen::MatrixXi(val);
    if (c * h * w != val.cols() * val.rows())
    {
        exit(-1);
    }
    this -> _val.resize(c * h * w, 1);
}

template<typename T>
Tensor<T>::Tensor(uint32_t c, uint32_t h, uint32_t w, std::string filepath)
{
    this -> c = c;
    this -> h = h;
    this -> w = w;
    T* arr = new T[c * h * w];
    std::ifstream dump_file;
    dump_file.open(filepath, std::ios::in | std::ios::binary);
    dump_file.read(reinterpret_cast<char*>(arr), sizeof(T) * c * h * w);
    dump_file.close();
    Eigen::MatrixXi mat(c * h * w, 1);
    this -> _val = mat;
    for (uint32_t i = 0; i < c * h * w; i++) {
        this -> _val(i, 0) = *(arr + i);
    }
    delete arr;
}

template<typename T>
Eigen::Matrix<T, -1, -1> Tensor<T>::to_matrix(uint32_t h, uint32_t w)
{
    Eigen::Matrix<T, -1, -1> mat(this -> _val);
    mat.resize(w, h);
    Eigen::Matrix<T, -1, -1> t_mat;
    t_mat = mat.transpose();
    return t_mat;
}

template<typename T>
void Tensor<T>::show()
{
    std::cout << "Tensor: ";
    std::cout << "c: " << this -> c;
    std::cout << ", h:" << this -> h;
    std::cout << ", w:" << this -> w;
    std::cout << ", val: [";
    // std::cout << ", val: [" << this -> _val(0) << ", ... ,";
    // std::cout << this -> _val(this -> c * this -> h * this -> w - 1);
    Eigen::MatrixXi mat(this -> _val);
    mat.resize(1, this -> c * this -> h * this -> w);
    std::cout << mat;
    std::cout << "]" << std::endl;
}

template<typename T>
bool Tensor<T>::operator==(Tensor<T> other)
{
    if (!(this -> c == other.c && this -> h == other.h && this -> w == other.w))
    {
        std::cout << "shape not equal" << std::endl;
        std::cout << "this chw:" << this -> c << " " << this -> h;
        std::cout << " " << this -> w << std::endl;
        std::cout << "other chw:" << other.c << " " << other.h;
        std::cout << " " << other.w << std::endl;
        return false;
    }
    for (uint32_t i = 0; i < this -> c * this -> h * this -> w; i++)
    {
        if (this -> _val(i) != other._val(i))
        {
            std::cout << "val[" << i << "] not equal" << std::endl;
            std::cout << "this: " << this -> _val(i);
            std::cout << " other: " << other._val(i) << std::endl;
            return false;
        }
    }
    return true;
}