# links

- blog 
https://intel-openvino-blog.webflow.io/blog-posts/custom-pytorch-operations

- repo
https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations

# Build
## build with cmake
```bash
mkdir build
cd build
OpenVINO_DIR=//openvino/runtime cmake ..
# or
cmake .. -DCMAKE_PREFIX_PATH=//openvino/runtime
# or
cmake .. -DCMAKE_FRAMEWORK_PATH=//openvino/runtime
# or
cmake .. -DCMAKE_APPBUNDLE_PATH=//openvino/runtime
```

## build with cmd
