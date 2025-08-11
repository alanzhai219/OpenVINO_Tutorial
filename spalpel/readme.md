# spalpel

## introduction

spalpel is an OpenVINO tool for cutting the snippet from the whole model.

## usage
```bash
python spalpel.py --model /mnt/disk/qwen3-8b/pytorch/dldt/compressed_weights/OV_FP16-4BIT_DEFAULT/openvino_model.xml --ins "__module.model.layers.0.self_attn.o_proj/ov_ext::linear/MatMul" --outs "__module.model.layers.0.self_attn.o_proj/ov_ext::linear/MatMul" --outpath "test/matmul_int4"
```

## next
⦁	support the multi ports as inputs
