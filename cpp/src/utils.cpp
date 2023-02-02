#include "utils.h"

std::string readFileIntoString(std::string filename)
{
    std::ifstream ifile(filename);
    std::ostringstream buf;
    char ch;
    while (buf && ifile.get(ch))
    {
        buf.put(ch);
    }
    return buf.str();
}

Eigen::MatrixXi create_matrix(cJSON *shape, cJSON *arr)
{
    std::vector<int32_t> shapes;
    // std::vector<int32_t> arr;
    cJSON *shape_item, *arr_item;
    cJSON_ArrayForEach(shape_item, shape)
    {
        shapes.push_back(shape_item -> valueint);
    }
    for (uint32_t i = 2; i < shapes.size(); i++)
    {
        if ((shapes[i] != 1))
        {
            // only kernel size one supported
            exit(-1);
        }
    }
    if (shapes.size() == 1)
    {
        shapes.push_back(1);
    }
    Eigen::MatrixXi matrix(shapes[0], shapes[1]);
    uint32_t matrix_idx=0;
    cJSON_ArrayForEach(arr_item, arr)
    {
        uint32_t col = matrix_idx % shapes[1];
        uint32_t row = matrix_idx / shapes[1];
        std::cout << "col:" << col << ", row:" << row << std::endl;
        matrix(row, col) = (arr_item -> valueint);
        matrix_idx ++;
    }
    return matrix;
}