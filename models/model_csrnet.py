import torch
import torch.nn as nn
import math
#from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .danet import DANetHead

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, channels, k=3, s=1, p=1, inplace=True):
        super(ConvBNReLU, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, channels, k, s, p, bias=False))
        self.add_module('bn', nn.BatchNorm2d(channels))
        self.add_module('relu', nn.ReLU(inplace=inplace))

class BasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
    
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).cuda())               
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut 

        return out

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channel = in_channels // 4
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + x)


class CSRNet(nn.Module):
    def __init__(self, crf, crf_channel, depth, alpha, num_classes, n_bands, avgpoosize, inplanes, psize, bottleneck=False):
        super(CSRNet, self).__init__()

        self.crf = crf
        self.crf_channel = crf_channel

        self.inplanes = inplanes
        self.input_featuremap_dim = self.inplanes
        # self.filter_set = nn.ModuleList([
        #     nn.Conv2d(in_channels=n_bands, out_channels=1, kernel_size=1, bias=False) for _ in range(crf_channel)
        # ])
        self.filter = nn.Conv2d(n_bands, crf_channel, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(crf_channel if self.crf else n_bands, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
        self.featuremap_dim = self.input_featuremap_dim 
        self.resnet = nn.ModuleList([BottleNeck(self.featuremap_dim, self.featuremap_dim) for _ in range(10)])
        self.head = DANetHead(self.featuremap_dim, self.featuremap_dim, nn.BatchNorm2d)
        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final= nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.tail1 = ConvBNReLU(self.final_featuremap_dim, self.final_featuremap_dim, k=3, s=2, p=1)
        self.tail2 = ConvBNReLU(self.final_featuremap_dim, self.final_featuremap_dim, k=3, s=2, p=0)
        self.avgpool = nn.AvgPool2d(avgpoosize)
        self.fc = nn.Linear(inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        if self.crf is True:
            # branch = []
            # for f in self.filter_set:
            #     branch.append(f(x))
            # x = torch.cat(branch, dim=1)
            x = self.filter(x)
        x = self.conv1(x)
        x = self.bn1(x)
        for conv in self.resnet:
            x = conv(x)
        x = self.head(x)

        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.tail1(x)
        x = self.tail2(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        self.feature = x.detach()
        x = self.fc(x)
        return x
