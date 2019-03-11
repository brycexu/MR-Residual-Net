import torch
import torch.nn as nn
import torch.nn.functional as F
from Binarized_Merge_and_Run_Structure2_Extended.utils import PreActBlock_MR, ModuleBinarizable, init_weight

class PreActResNet_MR(ModuleBinarizable):
    def __init__(self, block, num_classes=10):
        super(PreActResNet_MR, self).__init__()

        self.register_parameter('conv0', init_weight(64, 3, 3, 3))

        self.layer1_left = self.make_layer(block, 64, 64)
        self.layer1_right = self.make_layer(block, 64, 64)

        self.register_parameter('shortcut1', init_weight(128, 128, 1, 1))
        self.layer20_left = self.make_layer(block, 64, 128)
        self.layer20_right = self.make_layer(block, 64, 128)
        self.layer21_left = self.make_layer(block, 128, 128)
        self.layer21_right = self.make_layer(block, 128, 128)

        self.register_parameter('shortcut2', init_weight(256, 256, 1, 1))
        self.layer30_left = self.make_layer(block, 128, 256)
        self.layer30_right = self.make_layer(block, 128, 256)
        self.layer31_left = self.make_layer(block, 256, 256)
        self.layer31_right = self.make_layer(block, 256, 256)

        self.register_parameter('shortcut3', init_weight(512, 512, 1, 1))
        self.layer40_left = self.make_layer(block, 256, 512)
        self.layer40_right = self.make_layer(block, 256, 512)
        self.layer41_left = self.make_layer(block, 512, 512)
        self.layer41_right = self.make_layer(block, 512, 512)

        self.bn4 = nn.BatchNorm2d(512)
        self.register_parameter('fc', init_weight(num_classes, 512))
        self.bn5 = nn.BatchNorm1d(10)
        self.ls = nn.LogSoftmax(dim=1)

    def make_layer(self, block, in_planes, out_planes):
        layer = block(in_planes, out_planes)
        return nn.Sequential(layer)

    def forward(self, x):
        x = F.conv2d(x, self._get_weight('conv0'), stride=1, padding=1)

        shortcut = x
        left_out = self.layer1_left(x)
        right_out = self.layer1_right(x)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = F.conv2d(shortcut, self._get_weight('shortcut1'), stride=2, padding=0)
        left_out = self.layer20_left(left_out)
        right_out = self.layer20_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut
        shortcut = (left_out + right_out) * 1/2
        left_out = self.layer21_left(left_out)
        right_out = self.layer21_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = F.conv2d(shortcut, self._get_weight('shortcut2'), stride=2, padding=0)
        left_out = self.layer30_left(left_out)
        right_out = self.layer30_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut
        shortcut = (left_out + right_out) * 1/2
        left_out = self.layer31_left(left_out)
        right_out = self.layer31_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = F.conv2d(shortcut, self._get_weight('shortcut3'), stride=2, padding=0)
        left_out = self.layer40_left(left_out)
        right_out = self.layer40_right(right_out)
        shortcut = (left_out + right_out) * 1/2
        left_out = self.layer41_left(left_out)
        right_out = self.layer41_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        out = left_out + right_out + shortcut

        out = self.bn4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.linear(out, self._get_weight('fc', binarize=False))
        out = self.bn5(out)
        out = self.ls(out)
        return out

def PreActResNet21_MR():
    return PreActResNet_MR(PreActBlock_MR)