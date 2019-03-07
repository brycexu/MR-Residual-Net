import torch.nn as nn
import torch.nn.functional as F
from Merge_and_Run.utils import PreActBlock_MR

class PreActResNet_MR(nn.Module):
    def __init__(self, block, num_classes=10):
        super(PreActResNet_MR, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut_conv0 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer1_left = self.make_layer(block, 64, 128)
        self.layer1_right = self.make_layer(block, 64, 128)
        self.shortcut_conv1 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False)
        self.layer2_left = self.make_layer(block, 128, 256)
        self.layer2_right = self.make_layer(block, 128, 256)
        self.shortcut_conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False)
        self.layer3_left = self.make_layer(block, 256, 512)
        self.layer3_right = self.make_layer(block, 256, 512)
        self.shortcut_conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=2, padding=0, bias=False)
        self.layer4_left = self.make_layer(block, 512, 512)
        self.layer4_right = self.make_layer(block, 512, 512)
        self.linear = nn.Linear(512, num_classes)

    def make_layer(self, block, in_planes, out_planes):
        layer = block(in_planes, out_planes)
        return nn.Sequential(layer)

    def forward(self, x):
        x = self.conv1(x)

        shortcut = self.shortcut_conv0(x)
        left_out = self.layer1_left(x)
        right_out = self.layer1_right(x)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = (left_out + right_out) * 1/2
        shortcut = self.shortcut_conv1(shortcut)
        left_out = self.layer2_left(left_out)
        right_out = self.layer2_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = (left_out + right_out) * 1/2
        shortcut = self.shortcut_conv2(shortcut)
        left_out = self.layer3_left(left_out)
        right_out = self.layer3_right(right_out)
        left_out = left_out + shortcut
        right_out = right_out + shortcut

        shortcut = (left_out + right_out) * 1/2
        shortcut = self.shortcut_conv3(shortcut)
        left_out = self.layer4_left(left_out)
        right_out = self.layer4_right(right_out)
        out = left_out + right_out + shortcut

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def PreActResNet21_MR():
    return PreActResNet_MR(PreActBlock_MR)
