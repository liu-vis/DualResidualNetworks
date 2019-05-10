import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
import pietorch.N_modules as n_mods
from PIL import Image
from torchvision import models
from torch.autograd import Variable
from .N_modules import InsNorm
from . import se_nets

class cleaner(nn.Module):
    def __init__(self):
        super(cleaner, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # DuRBs, a DualUpDownLayer is a DuRB_US
        self.rud1 = DualUpDownLayer(64, 64, 64, f_size=5, dilation=1, norm_type='batch_norm')
        self.rud2 = DualUpDownLayer(64, 64, 64, f_size=5, dilation=1, norm_type='batch_norm')
        self.rud3 = DualUpDownLayer(64, 64, 64, f_size=7, dilation=1, norm_type='batch_norm')
        self.rud4 = DualUpDownLayer(64, 64, 64, f_size=7, dilation=1, norm_type='batch_norm')
        self.rud5 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud6 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud7 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud8 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud9 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud10 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud11 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')
        self.rud12 = DualUpDownLayer(64, 64, 64, f_size=11, dilation=1, norm_type='batch_norm')

        # Last layers
        # -- Up1 --
        self.upconv1 = ConvLayer(64, 256, kernel_size=3, stride=1)
        self.upsamp1 = nn.PixelShuffle(2)
        #----------
        self.conv4 = ConvLayer(64, 64, kernel_size=3, stride=1)

        # -- Up2 --
        self.upconv2 = ConvLayer(64, 256, kernel_size=3, stride=1)
        self.upsamp2 = nn.PixelShuffle(2)        
        #----------
        self.conv5 = ConvLayer(64, 64, kernel_size=3, stride=1)

        self.end_conv = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()        

        
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        res = x
        x = self.relu(self.conv3(x))

        x, res = self.rud1(x, res)
        x, res = self.rud2(x, res)
        x, res = self.rud3(x, res)
        x, res = self.rud4(x, res)
        x, res = self.rud5(x, res)
        x, res = self.rud6(x, res)
        x, res = self.rud7(x, res)
        x, res = self.rud8(x, res)
        x, res = self.rud9(x, res)
        x, res = self.rud10(x, res)
        x, res = self.rud11(x, res)
        x, res = self.rud12(x, res)

        x = self.upconv1(x)
        x = self.upsamp1(x)
        x = self.relu(self.conv4(x))

        x = self.upconv2(x)
        x = self.upsamp2(x)        
        x = self.relu(self.conv5(x))

        x = self.tanh(self.end_conv(x))
        x = x + residual
        
        return x


# DualUpDownLayer is DuRB_US, defined here:
class DualUpDownLayer(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, f_size=3, dilation=1, norm_type="instance", with_relu=True):
        super(DualUpDownLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        
        # T^{l}_{1}: (up+conv.)
        # -- Up --
        self.conv_pre = ConvLayer(in_dim, 4*in_dim, 3, 1)
        self.upsamp = nn.PixelShuffle(2)
        # --------
        self.up_conv = ConvLayer(res_dim, res_dim, kernel_size=f_size, stride=1, dilation=dilation)

        # T^{l}_{2}: (se+conv.), stride=2 for down-scaling.
        self.se = se_nets.SEBasicBlock(res_dim, res_dim, reduction=32)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=3, stride=2)

        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(x)

        x = self.conv_pre(x)
        x = self.upsamp(x)
        x = self.up_conv(x)
        x+= res
        x = self.relu(x)
        res = x

        x = self.se(x)
        x = self.down_conv(x)
        x+= x_r        

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res               


# ------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        self.dilation=dilation
        if dilation == 1:            
            reflect_padding = int(np.floor(kernel_size/2))
            self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, padding=dilation)

    def forward(self, x):
        if self.dilation == 1:
            out = self.reflection_pad(x)
            out = self.conv2d(out)
        else:
            out = self.conv2d(x)
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

