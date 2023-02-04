#include <iostream>
#include "quant_conv.h"
#include "tensor.h"
#include "utils.h"
#include "cJSON/cJSON.h"

#define CONV_JSON "../../py/dumps/conv.json"
#define CONV_INPUT "../../py/dumps/q_input_asint32.bin"
#define CONV_OUTPUT "../../py/dumps/q_output_asint32.bin"
#define CONV_INFO_JSON "../../py/dumps/quant_conv_info.json"

int main()
{
    std::string json_config_content = readFileIntoString(CONV_INFO_JSON);
    cJSON *quant_config_config = cJSON_Parse(json_config_content.c_str());
    cJSON *input_shape = cJSON_GetObjectItemCaseSensitive(quant_config_config, "input_shape");
    cJSON *output_shape = cJSON_GetObjectItemCaseSensitive(quant_config_config, "output_shape");
    std::vector<int32_t> input_shapes, output_shapes;
    cJSON *shape_item;
    cJSON_ArrayForEach(shape_item, input_shape)
    {
        input_shapes.push_back(shape_item -> valueint);
    }
    cJSON_ArrayForEach(shape_item, output_shape)
    {
        output_shapes.push_back(shape_item -> valueint);
    }
    QuantConv quant_conv(std::string(CONV_JSON));
    Tensor<int32_t> q_int32_in(input_shapes[1], input_shapes[2], input_shapes[3], CONV_INPUT);
    Tensor<int32_t> q_int32_out_gt(output_shapes[1], output_shapes[2], output_shapes[3], CONV_OUTPUT);
    Tensor<int32_t> q_int32_out = quant_conv.call(q_int32_in);
    // q_int32_in.show();
    std::cout << "calculated result:\n";
    q_int32_out.show();
    std::cout << "Ground Truth:\n";
    q_int32_out_gt.show();
    if (q_int32_out == q_int32_out_gt)
    {
        std::cout << "\033[032m[Consistency test passed]\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\033[031m[Consistency test failed]\033[0m" << std::endl;
    }
    return 0;
}