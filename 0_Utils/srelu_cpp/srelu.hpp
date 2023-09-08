#pragma once
#include <openvino/op/op.hpp>

namespace CustomExtension {

class SRelu : public ov::op::Op {
public:
    OPENVINO_OP("srelu_kernel"); // is the op name in the graph

    SRelu() = default;
    // SRelu(const ov::OutputVector &args, float weights_threshold_right, float weights_slope_right, float weights_threshold_left, float weights_slope_left);
    SRelu(const ov::OutputVector &args, float threshold, float alpha);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector &new_args) const override;
    bool visit_attributes(ov::AttributeVisitor &visitor) override;

    bool evaluate(ov::TensorVector &outputs, const ov::TensorVector &inputs) const override;
    bool has_evaluate() const override;

private:
    // float m_threshold_right;
    // float m_alpha_right;
    // float m_threshold_left;
    // float m_alpha_left;
    float m_threshold;
    float m_alpha;
};

} // namespace Extension
