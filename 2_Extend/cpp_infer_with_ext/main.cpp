#include <openvino/openvino.hpp>
#include <openvino/frontend/extension.hpp>
#include "complex_mul.hpp"

int main(int argc, char* agrv[]) {
    // 1. create openvino runtime core object
    ov::Core core;

    // 2. load extension
    // core.add_extension<TemplateExtension::ComplexMultiplication>();
    core.add_extension(ov::frontend::OpExtension<TemplateExtension::ComplexMultiplication>());

    // 3. compile model
    auto compiled_model = core.compile_model("../model_export/model_ver1.onnx");

    // 4. create request
    auto infer_request = compiled_model.create_infer_request();

    // 5. set input
    auto inputs = compiled_model.inputs();
    for (const auto i : inputs) {
        std::cout << i.get_any_name() << " - " << i.get_shape() << " - " << i.get_element_type() << "\n";
        ov::Tensor tensor(i.get_element_type(), i.get_shape());
        infer_request.set_tensor(i.get_any_name(), tensor);
    }

    // 6. do infer
    std::cout << "start infer ...\n";
    infer_request.infer();

    // 7. get output
    auto outputs = compiled_model.outputs();
    for (const auto i : outputs) {
        ov::Tensor out = infer_request.get_tensor(i.get_any_name());
        std::cout << i.get_any_name() << " - " << out.get_shape() << " - " << out.get_element_type() << "\n";
    }

    return 0;
}
