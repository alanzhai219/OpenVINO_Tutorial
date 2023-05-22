 #include <openvino/openvino.hpp>

 int main(int argc, char* argv[]) {
     // step 1: init OpenVINO Runtime Core
     ov::Core core;

     // step 2: read a model
     std::shared_ptr<ov::Model> model = core.read_model("bvlcalexnet-12.onnx");

     // step 3: compile a molde
     ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

     // step 4: create a tensor from image
     size_t element_nums = 1*3*224*224;
     float* in_ptr = static_cast<float*>(malloc(element_nums * sizeof(float)));
     ov::Tensor input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), in_ptr);

     // step 5: create a inference request
     ov::InferRequest infer_request = compiled_model.create_infer_request();
     infer_request.set_input_tensor(input_tensor);
     infer_request.infer();

     // step 6: retrieve inference results
     const ov::Tensor &output_tensor = infer_request.get_output_tensor();
     ov::Shape output_shape = output_tensor.get_shape();
     float* out_ptr = output_tensor.data<float>();

     return 0;
 }
