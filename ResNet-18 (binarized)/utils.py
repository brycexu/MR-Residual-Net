import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def init_weight(*args):
    return nn.Parameter(nn.init.kaiming_normal_(torch.zeros(*args), mode='fan_out', nonlinearity='relu'))

class ForwardSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return math.sqrt(2. / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    @staticmethod
    def backward(ctx, g):
        return g

class ModuleBinarizable(nn.Module):
    def __init__(self):
        super(ModuleBinarizable, self).__init__()

    def _get_weight(self, name, binarize=True):
        self.binarize = binarize
        w = getattr(self, name)
        return ForwardSign.apply(w) if self.binarize else w

    def forward(self, x):
        pass

class PreActBlock_MR(ModuleBinarizable):

    def __init__(self, in_planes, out_planes):
        super(PreActBlock_MR, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=False)
        self.register_parameter('conv1', init_weight(out_planes, in_planes, 3, 3))
        if in_planes == out_planes:
            self.conv1stride = 1
        else:
            self.conv1stride = 2
        self.bn2 = nn.BatchNorm2d(out_planes, affine=False)
        self.register_parameter('conv2', init_weight(out_planes, out_planes, 3, 3))

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = F.conv2d(out, self._get_weight('conv1'), stride=self.conv1stride, padding=1)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.conv2d(out, self._get_weight('conv2'), stride=1, padding=1)
        return out

