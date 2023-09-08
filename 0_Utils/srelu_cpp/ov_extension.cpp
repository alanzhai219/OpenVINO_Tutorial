#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>

#include "srelu.hpp"

// clang-format off
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({

        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<CustomExtension::SRelu>>(),

        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<CustomExtension::SRelu>>()
    }));
// clang-format on
