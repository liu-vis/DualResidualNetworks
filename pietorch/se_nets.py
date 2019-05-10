import torch
import torchvision
import math
import torch.nn as nn
from torchvision.models import ResNet
from .N_modules import SELayer
from .N_modules import InsNorm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# SE-ResNet Module    
class SEBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=64, with_norm=False):
        super(SEBasicBlock, self).__init__()
        self.with_norm = with_norm
        
        self.conv1 = conv3x3(inplanes, planes, stride)                    
        self.conv2 = conv3x3(planes, planes, 1)        
        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace=True)        
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.se(out)        
        out += x        
        out = self.relu(out)
        return out
