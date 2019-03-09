import torch.nn as nn
import torch.nn.functional as F

class PreActBlock_MR(nn.Module):

    def __init__(self, in_planes, out_planes):
        super(PreActBlock_MR, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        if in_planes == 64:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out
