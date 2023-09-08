import torch
import torch.nn as nn

class SReLU(nn.Module):
    def __init__(self, normalized_shape=(1,), threshold=6+1e-1, alpha=.2):
        super().__init__()

        # Cast to Tuple, whatever the original type
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        self.threshold_l = nn.Parameter(torch.full(normalized_shape, -threshold, requires_grad=True))
        self.threshold_r = nn.Parameter(torch.full(normalized_shape, +threshold, requires_grad=True))

        self.alpha_l = nn.Parameter(torch.full(normalized_shape, alpha, requires_grad=True))
        self.alpha_r = nn.Parameter(torch.full(normalized_shape, alpha, requires_grad=True))
        '''
        self.threshold_l = nn.Parameter(torch.full(normalized_shape, -0.001, requires_grad=True))
        self.threshold_r = nn.Parameter(torch.full(normalized_shape, +1., requires_grad=True))

        self.alpha_l = nn.Parameter(torch.full(normalized_shape, 0.001, requires_grad=True))
        self.alpha_r = nn.Parameter(torch.full(normalized_shape, 1., requires_grad=True))
        '''

    def forward(self, x):
        return torch.where(x > self.threshold_r, self.threshold_r + self.alpha_r * (x - self.threshold_r),
                           torch.where(x < self.threshold_l, self.threshold_l + self.alpha_r * (x - self.threshold_l), x))

class SReLU_Symbol(torch.autograd.Function):
    @staticmethod
    def forward(self, x: torch.Tensor, threshold: float=6+1e-1, alpha: float=.2) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return x

    @staticmethod
    def symbolic(g, x: torch.Tensor, threshold: float=6+1e-1, alpha: float=.2):
        y = g.op("srelu_kernel", x, thres_f = threshold, alpha_f = alpha)
        return y
