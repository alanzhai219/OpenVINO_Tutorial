- build with cmake
```bash
mkdir build
cd build
OpenVINO_DIR=//openvino/build/install/runtime cmake ..
make
```

- build with cmd
```bash
g++ complex_mul.cpp main.cpp -I//openvino/build/install/runtime/include -o main -L//openvino/build/install/runtime/lib/intel64 -lopenvino -ltbb -g
```
