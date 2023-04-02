cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(TARGET_NAME "complex_mul_extension")

find_package(OpenVINO REQUIRED)

set(SRC complex_mul.cpp ov_extension.cpp)

add_library(${TARGET_NAME} MODULE ${SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)