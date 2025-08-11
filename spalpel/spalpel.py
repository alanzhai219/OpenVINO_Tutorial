import argparse
from openvino.runtime import Core, serialize, Model
from openvino.runtime import opset13 as opset
from openvino.runtime import op
import openvino

def parse():
    parser = argparse.ArgumentParser(description="export graph to xml file")
    parser.add_argument("--model", type=str, required=True, help="model file, support openvino")
    parser.add_argument("--device", type=str, default="CPU", help="specify the device, support CPU/GPU")
    parser.add_argument("--ins", type=str, nargs="+", help="input nodes of new model file")
    parser.add_argument("--outs", type=str, nargs="+", help="output nodes of new model file")
    parser.add_argument("--outpath", type=str, default="out", help="specify the export xml name")
    args = parser.parse_args()
    return args

def main(args):
    ie = Core()
    model = ie.read_model(args.model)
    ops = model.get_ordered_ops()

    nodes_in = []
    for node in ops:
        if node.get_friendly_name() in args.ins:
            # 1. get the index for this node
            # 2. copy the info to init the parameter
            # 3. connect the parameter to the node's index input
            pname = node.get_friendly_name() + "_input"
            print("find input node {}".format(node.get_friendly_name()))

            idx = 0
            ptype = node.get_input_element_type(idx)
            pshape = node.get_input_partial_shape(idx)

            param = op.Parameter(ptype, pshape)
            param.set_friendly_name(pname)

            ins = node.inputs()[idx]

            # param -> output
            param_out = param.output(0)
            ins.replace_source_output(param_out)

            nodes_in.append(param)
            continue

    nodes_out = []
    for node in ops:
        if node.get_friendly_name() in args.outs:
            print("find output node {}".format(node.get_friendly_name()))
            for i in range(node.get_output_size()):
                result_op = opset.result(node.output(i))
                nodes_out.append(result_op)

    # cut_model = Model(nodes_out, model.get_parameters(), "model_snippet")
    cut_model = Model(nodes_out, nodes_in, "model_snippet")
    serialize(cut_model, "{}.xml".format(args.outpath), "{}.bin".format(args.outpath))

if __name__ == "__main__":
    args = parse()
    main(args)
