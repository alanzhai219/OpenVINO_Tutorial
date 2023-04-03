import torch
import torch.nn as nn

def complex_multiplication(input_tensor: torch.Tensor, other_tensor: torch.Tensor) -> torch.Tensor:
    complex_index = -1
    real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
    imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]
    multiplication = torch.cat(
      [
        real_part.unsqueeze(dim=complex_index),
        imaginary_part.unsqueeze(dim=complex_index),
      ],
      dim=complex_index,
    )
    return multiplication

class ComplexMul_ver0(nn.Module):
    def __init__(self):
      super(ComplexMul_ver0, self).__init__()

    def forward(self, x, y):
      return complex_multiplication(x, y)

class ComplexMul_ver1(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_tensor, other_tensor, is_conj = True):
        return g.op("ComplexMultiplication", input_tensor, other_tensor, is_conj_i=int(is_conj))

    @staticmethod
    def forward(self, input_tensor, other_tensor):
        multiplication = complex_multiplication(input_tensor, other_tensor)
        '''
        complex_index = -1
        real_part = input_tensor[..., 0] * other_tensor[..., 0] - input_tensor[..., 1] * other_tensor[..., 1]
        imaginary_part = input_tensor[..., 0] * other_tensor[..., 1] + input_tensor[..., 1] * other_tensor[..., 0]

        multiplication = torch.cat(
            [
                real_part.unsqueeze(dim=complex_index),
                imaginary_part.unsqueeze(dim=complex_index),
            ],
            dim=complex_index,
        )
        '''
        return multiplication

