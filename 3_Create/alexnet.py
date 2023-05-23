import os
import struct
import numpy as np

from openvino.runtime import (Core, Shape, Model, op, Type, serialize)
from openvino.runtime import opset10 as default_opset

import pdb

BATCH_SIZE = 1
INPUT_H = 224
INPUT_W = 224
OUTPUT_SIZE = 1000

WEIGHT_PATH = "./alexnet.wts"
def load_weights(file):
    print(f"Loading weights: {file}")
    assert(os.path.exists(file), "Unable to load weight file")

    weight_map = {}
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
    count = int(lines[0])
    assert(count == len(lines) - 1)
    for i in range(1, count + 1):
        splits = lines[i].split(" ")
        name = splits[0]
        cur_count = int(splits[1])
        assert(cur_count + 2 == len(splits))
        values = []
        for j in range(2, len(splits)):
            values.append(struct.unpack(">f", bytes.fromhex(splits[j])))
        weight_map[name] = np.array(values, dtype=np.float32)
        print("key={}, shape={}".format(name, weight_map[name].shape))

    return weight_map

def create_model():
    weight_map = load_weights(WEIGHT_PATH)

    # input
    input_shape = [BATCH_SIZE, 3, INPUT_H, INPUT_W]
    input_param = op.Parameter(Type.f32, Shape(input_shape))
    input_param.set_friendly_name("image")

    # conv1
    kernel1 = op.Constant(Type.f32, Shape([64,3,11,11]), weight_map["features.0.weight"])
    conv1 = default_opset.convolution(data=input_param,
                                      filters=kernel1,
                                      strides=[4,4],
                                      pads_begin=[2,2],
                                      pads_end=[2,2],
                                      dilations=[1,1])

    # add1
    bias1 = op.Constant(Type.f32, Shape([1,64,1,1]), weight_map["features.0.bias"])
    add1 = default_opset.add(conv1,
                             bias1)

    # relu1
    relu1 = default_opset.relu(add1)

    # pool1
    pool1 = default_opset.max_pool(data=relu1,
                                   dilations=[1,1],
                                   strides=[2,2],
                                   pads_begin=[0,0],
                                   pads_end=[0,0],
                                   kernel_shape=[3,3])

    # conv2
    kernel2 = op.Constant(Type.f32, Shape([192,64,5,5]), weight_map["features.3.weight"])
    conv2 = default_opset.convolution(data=pool1.output(0),
                                      filters=kernel2,
                                      strides=[1,1],
                                      pads_begin=[2,2],
                                      pads_end=[2,2],
                                      dilations=[1,1])
    # add2
    bias2 = op.Constant(Type.f32, Shape([1,192,1,1]), weight_map["features.3.bias"])
    add2 = default_opset.add(conv2,
                             bias2)

    # relu2
    relu2 = default_opset.relu(add2)

    # pool2
    pool2 = default_opset.max_pool(data=relu2,
                                   dilations=[1,1],
                                   strides=[2,2],
                                   pads_begin=[0,0],
                                   pads_end=[0,0],
                                   kernel_shape=[3,3])

    # conv3
    kernel3 = op.Constant(Type.f32, Shape([384,192,3,3]), weight_map["features.6.weight"])
    conv3 = default_opset.convolution(data=pool2.output(0),
                                      filters=kernel3,
                                      strides=[1,1],
                                      pads_begin=[1,1],
                                      pads_end=[1,1],
                                      dilations=[1,1])
    assert conv3

    # add3
    bias3 = op.Constant(Type.f32, Shape([1,384,1,1]), weight_map["features.6.bias"])
    add3 = default_opset.add(conv3,
                             bias3)
    assert add3

    # relu3
    relu3 = default_opset.relu(add3)
    assert relu3

    # conv4
    kernel4 = op.Constant(Type.f32, Shape([256,384,3,3]), weight_map["features.8.weight"])
    conv4 = default_opset.convolution(data=relu3,
                                      filters=kernel4,
                                      strides=[1,1],
                                      pads_begin=[1,1],
                                      pads_end=[1,1],
                                      dilations=[1,1])
    assert conv4
    # add4
    bias4 = op.Constant(Type.f32, Shape([1,256,1,1]), weight_map["features.8.bias"])
    add4 = default_opset.add(conv4,
                             bias4)
    assert add4

    # relu4
    relu4 = default_opset.relu(add4)
    assert relu4

    # conv5
    kernel5 = op.Constant(Type.f32, Shape([256,256,3,3]), weight_map["features.10.weight"])
    conv5 = default_opset.convolution(data=relu4,
                                      filters=kernel5,
                                      strides=[1,1],
                                      pads_begin=[1,1],
                                      pads_end=[1,1],
                                      dilations=[1,1])
    # add5
    bias5 = op.Constant(Type.f32, Shape([1,256,1,1]), weight_map["features.10.bias"])
    add5 = default_opset.add(conv5,
                             bias4)

    # relu5
    relu5 = default_opset.relu(add5)

    # pool5
    pool5 = default_opset.max_pool(data=relu5,
                                   dilations=[1,1],
                                   strides=[2,2],
                                   pads_begin=[0,0],
                                   pads_end=[0,0],
                                   kernel_shape=[3,3])

    # fc1
    fc_kernel1 = op.Constant(Type.f32, Shape([4096, 9216]), weight_map["classifier.1.weight"])
    fc1 = default_opset.matmul(data_a=default_opset.reshape(pool5.output(0),
                                                            op.Constant(Type.i64, Shape([2]), [1, 9216]),
                                                            True),
                               data_b=fc_kernel1,
                               transpose_a=False,
                               transpose_b=True)
    assert fc1

    # fc-add1
    fc_bias1 = op.Constant(Type.f32, Shape([1, 4096]), weight_map["classifier.1.bias"])
    fc_add1 = default_opset.add(fc1,
                                fc_bias1)
    # relu6
    relu6 = default_opset.relu(fc_add1)

    # fc2
    fc_kernel2 = op.Constant(Type.f32, Shape([4096, 4096]), weight_map["classifier.4.weight"])
    fc2 = default_opset.matmul(data_a=default_opset.reshape(relu6,
                                                            op.Constant(Type.i64, Shape([2]), [1,4096]),
                                                            True),
                               data_b=fc_kernel2,
                               transpose_a=False,
                               transpose_b=True)
    assert fc2

    # add7
    fc_bias2 = op.Constant(Type.f32, Shape([1, 4096]), weight_map["classifier.4.bias"])
    fc_add2 = default_opset.add(fc2,
                                fc_bias2)

    # relu7
    relu7 = default_opset.relu(fc_add2)

    # fc3
    fc_kernel3 = op.Constant(Type.f32, Shape([1000, 4096]), weight_map["classifier.6.weight"])
    fc3 = default_opset.matmul(data_a=default_opset.reshape(relu7,
                                                            op.Constant(Type.i64, Shape([2]), [1,4096]),
                                                            True),
                               data_b=fc_kernel3,
                               transpose_a=False,
                               transpose_b=True)
    assert fc3

    fc_bias3 = op.Constant(Type.f32, Shape([1, 1000]), weight_map["classifier.6.bias"])
    fc_add3 = default_opset.add(fc3,
                                fc_bias3)

    model = Model(fc3, [input_param], "AlexNet")
    return model

def do_inference():
    pass
if __name__ == "__main__":
    model = create_model()
    serialize(model, "alexnet.xml", "alexnet.bin")

