- compile with cmd
```bash
# for release
g++ main.cpp -I//install_toolkit/runtime/include -o main -L//install_toolkit/runtime/lib/intel64 -lopenvinog
# for debug
g++ main.cpp -I//install_toolkit/runtime/include -o main -L//install_toolkit/runtime/lib/intel64 -lopenvino -g
```

- compile with cmake

create a `CMakeLists.txt`
```cmake
add_executable(main main.cpp)

target_link_libraries(main
//install_toolkit/runtime/lib/intel64/libopenvino.so
)

target_include_directories(main PUBLIC
//install_toolkit/runtime/include/
# //install_toolkit/runtime/include/ie
)
```
compile
```bash
mkdir build
cd build
cmake ..
```
