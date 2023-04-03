# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import argparse
import torch
import torch.nn as nn
# from torch.autograd import Variable
from complex_mul import ComplexMul_ver0, ComplexMul_ver1

class MyModel_ver0(nn.Module):
    def __init__(self):
        super(MyModel_ver0, self).__init__()
        self.complex_mul = ComplexMul_ver0()

    def forward(self, x, y):
        return self.complex_mul(x, y)

class MyModel_ver1(nn.Module):
    def __init__(self):
        super(MyModel_ver1, self).__init__()
        self.complex_mul = ComplexMul_ver1()

    def forward(self, x, y):
        return self.complex_mul.apply(x, y)

def export(model, export_name, inp_shape=[3, 2, 4, 8, 2], other_shape=[3, 2, 4, 8, 2]):
    np.random.seed(324)
    torch.manual_seed(32)

    inp0 = torch.randn(inp_shape)
    inp1 = torch.randn(other_shape)
    model.eval()

    with torch.no_grad():
        torch.onnx.export(model, (inp0, inp1), export_name,
                          input_names=['input0', 'input1'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    ref = model(inp0, inp1)
    return [inp0.detach().numpy(), inp1.detach().numpy()], ref.detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ONNX model and test data')
    parser.add_argument('--inp_shape', type=int, nargs='+', default=[3, 2, 4, 8, 2])
    parser.add_argument('--other_shape', type=int, nargs='+', default=[3, 2, 4, 8, 2])
    args = parser.parse_args()

    model = MyModel_ver0()
    export(model, "model_ver0.onnx", args.inp_shape, args.other_shape)

    model = MyModel_ver1()
    export(model, "model_ver1.onnx", args.inp_shape, args.other_shape)
