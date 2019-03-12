import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import PreActBlock_MR, ModuleBinarizable, init_weight

class PreActResNet_MR(ModuleBinarizable):
    def __init__(self, block, num_classes=10):
        super(PreActResNet_MR, self).__init__()

        self.register_parameter('conv0', init_weight(64, 3, 3, 3))

        self.layer1_left = self.make_layer(block, 64, 64)
        self.layer1_right = self.make_layer(block, 64, 64)

        self.layer2_left = self.make_layer(block, 64, 128)
        self.layer2_right = self.make_layer(block, 64, 128)

        self.layer3_left = self.make_layer(block, 128, 256)
        self.layer3_right = self.make_layer(block, 128, 256)

        self.layer4_left = self.make_layer(block, 256, 512)
        self.layer4_right = self.make_layer(block, 256, 512)

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
        shortcut = F.avg_pool2d(shortcut, 2)
        left_out = self.layer2_left(left_out)
        right_out = self.layer2_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = F.avg_pool2d(shortcut, 2)
        left_out = self.layer3_left(left_out)
        right_out = self.layer3_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = torch.cat([left_out, right_out], dim=1)
        shortcut = F.avg_pool2d(shortcut, 2)
        left_out = self.layer4_left(left_out)
        right_out = self.layer4_right(right_out)
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
