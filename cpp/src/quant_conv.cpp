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
    std::cout << json_config_content << std::endl;
}

void QuantConv::show() {
    std::cout << "[quant conv]:";
    std::cout << " | (weight): shift:" << this -> weight_shiftbit;
    std::cout << ", num bit:" << this -> weight_num_bit;
    std::cout << " | (bias): shift:" << this -> bias_shiftbit;
    std::cout << ", num bit:" << this -> bias_num_bit << std::endl;
    std::cout << "weight: " << this -> weight << std::endl;
    std::cout << "bias: " << this -> bias << std::endl;
}

Eigen::MatrixXi QuantConv::call(
    Eigen::MatrixXi input_i,
    uint32_t c,
    uint32_t h,
    uint32_t w) {
        input_i.resize(c, h * w);
        Eigen::MatrixXi mul_result = this -> weight * input_i;
        mul_result.colwise() += this -> bias;
        return mul_result;
}