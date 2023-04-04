cmake_minimum_required(VERSION 3.23)
project(DEMO)
set(CMAKE_CXX_STANDARD 11)

set(TARGET_NAME "complex_mul")

find_package(OpenVINO REQUIRED)

set(SRC main.cpp complex_mul.cpp)

add_executable(${TARGET_NAME} ${SRC})

# target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime tbb)
