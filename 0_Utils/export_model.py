from openvino.tools.mo import convert_model
from openvino.runtime import save_model
import torch

# export native alexnet
from model_hack.alexnet import AlexNet
alexnet_model = AlexNet()
alexnet_model.load_state_dict(torch.load('checkpoints/alexnet-model.pth'))
# script_model = torch.jit.script(alexnet_model)

ov_model = convert_model(alexnet_model)
save_model(ov_model, "ov_model/native/alexnet.xml")

with torch.no_grad():
    dummy_input = torch.randn(1,3,224,224)
    torch.onnx.export(
            model=alexnet_model,
            args=dummy_input,
            f="ov_model/native/alexnet_native.onnx",
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

# export srelu alexnet
from model_hack.alexnet_srelu import AlexNet
alexnet_model = AlexNet()
alexnet_model.load_state_dict(torch.load('checkpoints/alexnet-model.pth'), strict=False)

ov_model = convert_model(alexnet_model)
save_model(ov_model, "ov_model/srelu/alexnet.xml")

with torch.no_grad():
    dummy_input = torch.randn(1,3,224,224)
    torch.onnx.export(
            model=alexnet_model,
            args=dummy_input,
            f="ov_model/srelu/alexnet_srelu.onnx",
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

# export onnx srelu alexnet
from model_hack.alexnet_srelu_symbol import AlexNet
alexnet_model = AlexNet()
alexnet_model.load_state_dict(torch.load('checkpoints/alexnet-model.pth'), strict=False)

with torch.no_grad():
    dummy_input = torch.randn(1,3,224,224)
    torch.onnx.export(
            model=alexnet_model,
            args=dummy_input,
            f="ov_model/onnx/alexnet_srelu_symbol.onnx",
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
'''
ov_model = convert_model(alexnet_model)
save_model(ov_model, "model/onnx/alexnet.xml")
'''
