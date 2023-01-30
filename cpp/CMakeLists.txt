cmake_minimum_required(VERSION 3.0)
project(naive_ptq)

# find package
find_package(Eigen3)
add_subdirectory(third_party/cJSON)

# add src directory
aux_source_directory(src NAIVE_PTQ_SRC)

# add library
add_library(${PROJECT_NAME} SHARED ${NAIVE_PTQ_SRC})

# add exec
add_executable(naive_ptq_bin app/naive_ptq_bin.cpp)

# include
target_include_directories(${PROJECT_NAME} PRIVATE third_party)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_include_directories(${PROJECT_NAME} PUBLIC ${EIGEN3_INCLUDE_DIRS})

# link
target_link_libraries(${PROJECT_NAME} ${CJSON_LIB})
target_link_libraries(naive_ptq_bin ${PROJECT_NAME})