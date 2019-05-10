import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
from PIL import Image
from torchvision import models
from torch.autograd import Variable

class cleaner(nn.Module):
    def __init__(self):
        super(cleaner, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = ConvLayer(3, 32, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(32, 32, kernel_size=3, stride=1)

        # DuRBs
        self.block1 = DuRB_p(k1_size=5, k2_size=3, dilation=1)
        self.block2 = DuRB_p(k1_size=7, k2_size=5, dilation=1)
        self.block3 = DuRB_p(k1_size=7, k2_size=5, dilation=2)
        self.block4 = DuRB_p(k1_size=11, k2_size=7, dilation=2)
        self.block5 = DuRB_p(k1_size=11, k2_size=5, dilation=1)
        self.block6 = DuRB_p(k1_size=11, k2_size=7, dilation=3)

        # Last layers
        self.conv3 = ConvLayer(32, 32, kernel_size=3, stride=1)
        self.conv4 = ConvLayer(32, 3, kernel_size=3, stride=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):        
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        res = out
        
        out, res = self.block1(out, res)
        out, res = self.block2(out, res)
        out, res = self.block3(out, res)
        out, res = self.block4(out, res)
        out, res = self.block5(out, res)
        out, res = self.block6(out, res)

        out = self.relu(self.conv3(out))
        out = self.tanh(self.conv4(out))
        out = out + x

        return out        

        
class DuRB_p(nn.Module):     
    def __init__(self, in_dim=32, out_dim=32, res_dim=32, k1_size=3, k2_size=1, dilation=1, norm_type="batch_norm", with_relu=True):
        super(DuRB_p, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        
        # T^{l}_{1}: (conv.)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)

        # T^{l}_{2}: (conv.)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=1)

        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(x)
        
        # T^{l}_{1}
        x = self.up_conv(x)
        x+= res
        x = self.relu(x)
        res = x

        # T^{l}_{2}
        x = self.down_conv(x)
        x+= x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass
            
        return x, res
       

#---------------------------------------------------------        
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)        
        return out

        
class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect.")

    def forward(self, x):
        out = self.norm(x)        
        return out
