#define QUANT_CONV_DEBUG

#include "quant_conv.h"
#include "cJSON/cJSON.h"

QuantConv::QuantConv(std::string json_config) {
    std::string json_config_content = readFileIntoString(json_config);
    cJSON *quant_config_config = cJSON_Parse(json_config_content.c_str());
    cJSON *weight = cJSON_GetObjectItemCaseSensitive(quant_config_config, "weight");
    cJSON *bias = cJSON_GetObjectItemCaseSensitive(quant_config_config, "bias");
    cJSON *input_shiftbit = cJSON_GetObjectItemCaseSensitive(
        quant_config_config, "input_shiftbit");
    this -> input_shiftbit = input_shiftbit -> valueint;
    cJSON *weight_num_bit = cJSON_GetObjectItemCaseSensitive(weight, "num_bit");
    cJSON *weight_shiftbit = cJSON_GetObjectItemCaseSensitive(weight, "shift_bit");
    cJSON *weight_shape = cJSON_GetObjectItemCaseSensitive(weight, "shape");
    cJSON *weight_arr = cJSON_GetObjectItemCaseSensitive(weight, "arr");
    this -> weight_shiftbit = weight_shiftbit -> valueint;
    this -> weight_num_bit = weight_num_bit -> valueint;
    this -> weight = create_matrix(weight_shape, weight_arr);
    cJSON *bias_num_bit = cJSON_GetObjectItemCaseSensitive(bias, "num_bit");
    cJSON *bias_shiftbit = cJSON_GetObjectItemCaseSensitive(bias, "shift_bit");
    cJSON *bias_shape = cJSON_GetObjectItemCaseSensitive(bias, "shape");
    cJSON *bias_arr = cJSON_GetObjectItemCaseSensitive(bias, "arr");
    this -> bias_shiftbit = bias_shiftbit -> valueint;
    this -> bias_num_bit = bias_num_bit -> valueint;
    this -> bias = create_matrix(bias_shape, bias_arr);
    this -> get_c();
}

void QuantConv::show() {
    std::cout << "[quant conv]:";
    std::cout << " | (weight): shift:" << this -> weight_shiftbit;
    std::cout << ", num bit:" << this -> weight_num_bit;
    std::cout << " | (bias): shift:" << this -> bias_shiftbit;
    std::cout << ", num bit:" << this -> bias_num_bit;
    std::cout << ", input_c:" << this -> input_c;
    std::cout << ", output_c:" << this -> output_c << std::endl;
    std::cout << "weight: " << this -> weight << std::endl;
    std::cout << "bias: " << this -> bias << std::endl;
}

void QuantConv::get_c() {
    this -> input_c = this -> weight.cols();
    this -> output_c = this -> weight.rows();
}

Tensor<int32_t> QuantConv::call(Tensor<int32_t> input) {
        Eigen::MatrixXi input_mat = input.to_matrix(input.c, input.h * input.w);
        Eigen::MatrixXi mul_result = this -> weight * input_mat;
#ifdef QUANT_CONV_DEBUG
        std::cout << "input mat: [" << input_mat.rows() << ", " << input_mat.cols() << "]\n";
        std::cout << input_mat << "\n";
        std::cout << "mul_result: [" << mul_result.rows() << ", " << mul_result.cols() << "]\n";
        std::cout << mul_result << "\n";
        std::cout << "bias shape: " << this -> bias.rows() << std::endl;
#endif
        mul_result.colwise() += this -> bias;
        return Tensor<int32_t>(this -> output_c, input.h, input.w, mul_result);
}