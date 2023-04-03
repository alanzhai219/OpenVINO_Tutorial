// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/node_context.hpp>

#include "complex_mul.hpp"

OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
       std::make_shared<ov::OpExtension<TemplateExtension::ComplexMultiplication>>(),
       std::make_shared<ov::frontend::OpExtension<TemplateExtension::ComplexMultiplication>>()
}));
