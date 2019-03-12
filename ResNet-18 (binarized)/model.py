import torch.nn as nn
import torch.nn.functional as F
from utils import PreActBlock_MR, ModuleBinarizable, init_weight

class PreActResNet_MR(ModuleBinarizable):
    def __init__(self, block, num_classes=10):
        super(PreActResNet_MR, self).__init__()

        self.register_parameter('conv0', init_weight(64, 3, 3, 3))

        self.layer1 = self.make_layer(block, 64, 64)

        self.layer20 = self.make_layer(block, 64, 128)
        self.register_parameter('shortcut1', init_weight(128, 64, 1, 1))
        self.layer21 = self.make_layer(block, 128, 128)

        self.layer30 = self.make_layer(block, 128, 256)
        self.register_parameter('shortcut2', init_weight(256, 128, 1, 1))
        self.layer31 = self.make_layer(block, 256, 256)

        self.layer40 = self.make_layer(block, 256, 512)
        self.register_parameter('shortcut3', init_weight(512, 256, 1, 1))
        self.layer41 = self.make_layer(block, 512, 512)

        self.bn4 = nn.BatchNorm2d(512)
        self.register_parameter('fc', init_weight(num_classes, 512))
        self.bn5 = nn.BatchNorm1d(10)
        self.ls = nn.LogSoftmax(dim=1)


    def make_layer(self, block, in_planes, out_planes):
        layer = block(in_planes, out_planes)
        return nn.Sequential(layer)

    def forward(self, x):
        out = F.conv2d(x, self._get_weight('conv0'), stride=1, padding=1)

        shortcut0 = out
        out = self.layer1(out)
        out += shortcut0

        shortcut1 = F.conv2d(out, self._get_weight('shortcut1'), stride=2, padding=0)
        out = self.layer20(out)
        out += shortcut1
        shortcut1 = out
        out = self.layer21(out)
        out += shortcut1

        shortcut2 = F.conv2d(out, self._get_weight('shortcut2'), stride=2, padding=0)
        out = self.layer30(out)
        out += shortcut2
        shortcut2 = out
        out = self.layer31(out)
        out += shortcut2

        shortcut3 = F.conv2d(out, self._get_weight('shortcut3'), stride=2, padding=0)
        out = self.layer40(out)
        out += shortcut3
        shortcut3 = out
        out = self.layer41(out)
        out += shortcut3

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.linear(out, self._get_weight('fc', binarize=False))
        out = self.bn5(out)
        out = self.ls(out)
        return out

def PreActResNet21_MR():
    return PreActResNet_MR(PreActBlock_MR)
