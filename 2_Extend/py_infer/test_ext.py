import openvino.runtime as ov

core = ov.Core()

# load extension library
core.add_extension("cpp_extend/build/libcomplex_mul_extension.so")

# load model
model = core.read_model("model_export/model.onnx")

print("done")
