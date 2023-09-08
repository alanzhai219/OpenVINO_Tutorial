#include "srelu.hpp"

using namespace CustomExtension;

// SRelu::SRelu(const ov::OutputVector &args, float weights_threshold_right, float weights_slope_right, float weights_threshold_left, float weights_slope_left)
//     : Op(args)
//     , m_threshold_right(weights_threshold_right)
//     , m_alpha_right(weights_slope_right)
//     , m_threshold_left(weights_threshold_left)
//     , m_alpha_left(weights_slope_left) {
//     constructor_validate_and_infer_types();
// }

// SRelu::SRelu(const ov::OutputVector &args, float threshold, float alpha)
//     : Op(args)
//     , m_threshold_right(threshold)
//     , m_alpha_right(alpha)
//     , m_threshold_left(threshold)
//     , m_alpha_left(alpha) {
//     constructor_validate_and_infer_types();
// }

// OP constructor
SRelu::SRelu(const ov::OutputVector &args, float threshold, float alpha)
    : Op(args)
    , m_threshold(threshold)
    , m_alpha(alpha) {
    std::cout << "## threshold: " << m_threshold << "\n";
    std::cout << "## alpha: " << m_alpha << "\n";
    constructor_validate_and_infer_types();
}

void SRelu::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<ov::Node>
SRelu::clone_with_new_inputs(const ov::OutputVector &new_args) const {
    return std::make_shared<SRelu>(new_args, m_threshold, m_alpha);
}

bool SRelu::visit_attributes(ov::AttributeVisitor &visitor) {
    // should map from onnx/pytorch attributes to openvino ir
    visitor.on_attribute("thres", m_threshold);
    visitor.on_attribute("alpha", m_alpha);
    // visitor.on_attribute("thres", m_threshold_left);
    // visitor.on_attribute("alpha", m_alpha_left);
    return true;
}

// compute kernel
bool SRelu::evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const {
    // ov::TensorVector is a vector of ov::Tensor, so can access by index.
    auto in_tensor = inputs[0];
    auto out_tensor = outputs[0];
    // get data pointer
    const float* inp = reinterpret_cast<float*>(inputs[0].data());
    float* out = reinterpret_cast<float*>(outputs[0].data());

    // auto f = [&](float x){
    //     if (x < this->m_threshold_left) {
    //         return x * this->m_alpha_left;
    //     } else if (x > this->m_threshold_right) {
    //         return x * this->m_alpha_right;
    //     } else {
    //         return x;
    //     }
    // };

    const auto elem_nums = inputs[0].get_size();
    for (size_t i=0; i < elem_nums; ++i) {
        if (inp[i] > m_threshold) {
            out[i] = m_alpha * inp[i];
        } else if (inp[i] < -m_threshold) {
            out[i] = m_alpha * inp[i];
        } else {
            out[i] = inp[i];
        }
        // out[i] = inp[i];
    }

    out_tensor.set_shape(in_tensor.get_shape());

    /*
    auto in = inputs[0];
    auto out = outputs[0];
    if (out.data() == in.data()) // Nothing to do
        return true;
    out.set_shape(in.get_shape());
    memcpy(out.data(), in.data(), in.get_byte_size());
    */
    // out_tensor.set_shape(in_tensor.get_shape());
    // memcpy(out_tensor.data(), in_tensor.data(), in_tensor.get_byte_size());
    return true;
}

bool SRelu::has_evaluate() const
{
    return true;
}

