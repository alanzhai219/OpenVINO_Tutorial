import os
import argparse
from openvino.runtime import Core, serialize, Model
from openvino.runtime import opset13 as opset
from openvino.runtime import op
import openvino

def parse():
    parser = argparse.ArgumentParser(description="export graph to xml file")
    parser.add_argument("--model", type=str, required=True, help="model file, support openvino")
    parser.add_argument("--device", type=str, default="CPU", help="specify the device, support CPU/GPU")
    parser.add_argument("--ins", type=str, help="input nodes of new model file. Like --ins \"name1@0,1,2; name2@3,4; name3@5,6\"")
    parser.add_argument("--outs", type=str, nargs="+", help="output nodes of new model file")
    parser.add_argument("--outpath", type=str, default="out", help="specify the export xml name")
    args = parser.parse_args()
    return args


class args_pack:
    def __init__(self, args):
        self.ori_model = args.model
        self.device = args.device
        self.outpath = args.outpath
        self.ins = args.ins
        self.outs = args.outs
        self.input_port_map = self.parse_dict_string(args.ins)
        # self.output_port_map = self.parse_dict_string(args.outs)

    def parse_dict_string(self, dict_str):
        result = {}
        groups = dict_str.split(';')
        for group in groups:
            group = group.strip()
            if not group:
                continue
            key, values = group.split('@')
            result[key.strip()] = [int(v) for v in values.split(',')]
        return result

def main(args):
    ie = Core()
    model = ie.read_model(args.ori_model)
    ops = model.get_ordered_ops()

    nodes_in = []
    for node in ops:
        # TODO: support the cut from the middle of the model
        if node.get_friendly_name() in args.input_port_map:

            # 1. get the index for this node
            # 2. copy the info to init the parameter
            # 3. connect the parameter to the node's index input
            #    use node.set_argument api
            found_name = node.get_friendly_name()
            pname = found_name + "_scalpel"
            print("find input node {}".format(found_name))
            
            found_ports = args.input_port_map[found_name]
            for idx in found_ports:
                ptype = node.get_input_element_type(idx)
                pshape = node.get_input_partial_shape(idx)
                
                param = op.Parameter(ptype, pshape)
                param.set_friendly_name(pname + "_" + str(idx))

                ins = node.inputs()[idx]

                # param -> output
                param_out = param.output(0)
                ins.replace_source_output(param_out)

                nodes_in.append(param)

    nodes_out = []
    for node in ops:
        if node.get_friendly_name() in args.outs:
            found_name = node.get_friendly_name()
            print("find output node {}".format(found_name))
            for i in range(node.get_output_size()):
                result_op = opset.result(node.output(i))
                nodes_out.append(result_op)

    spalpel_model = Model(nodes_out, nodes_in, "model_scalpel")
    # cut_model = Model(nodes_out, model.get_parameters(), "model_snippet")

    ori_model_name = os.path.splitext(os.path.basename(args.ori_model))[0]
    spalpel_model_name = "scalpel_" + ori_model_name
    save_model_name = os.path.join(args.outpath, spalpel_model_name)
    serialize(spalpel_model, "{}.xml".format(save_model_name), "{}.bin".format(save_model_name))

if __name__ == "__main__":
    args = parse()
    args_package = args_pack(args)
    main(args_package)
