#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <Eigen/Core>
#include "cJSON/cJSON.h"

std::string readFileIntoString(std::string filename);
Eigen::MatrixXi create_matrix(cJSON *shape, cJSON *arr);

