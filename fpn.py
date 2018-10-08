# NOTE! based off of https://github.com/kuangliu/pytorch-retinanet/blob/master/fpn.py and https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        super(Bottleneck, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w))
        out = F.relu(self.bn2(self.conv2(out))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn3(self.conv3(out)) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, out_channels, h, w) if stride == 1, (batch_size, out_channels, h/2, w/2) if stride == 2)

        return out

class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.in_channels = 64

        self.layer2 = self._make_layer(block, channels=64, num_blocks=num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, channels=128, num_blocks=num_blocks[1], stride=2)
        self.layer4 = self._make_layer(block, channels=256, num_blocks=num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, channels=512, num_blocks=num_blocks[3], stride=2)

        self.conv6 = nn.Conv2d(block.expansion*512, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.lateral_conv5 = nn.Conv2d(block.expansion*512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv4 = nn.Conv2d(block.expansion*256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(block.expansion*128, 256, kernel_size=1, stride=1, padding=0)

        self.smoothing_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothing_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

        blocks = []
        for stride in strides:
            blocks.append(block(in_channels=self.in_channels, channels=channels, stride=stride))
            self.in_channels = block.expansion*channels

        layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

        return layer

    def _upsample_and_add(self, feature_map, small_feature_map):
        # (feature_map has shape (batch_size, channels, h, w))
        # (small_feature_map has shape (batch_size, channels, h/2, w/2)) (integer division)

        _, _, h, w = feature_map.size()

        out = F.upsample(small_feature_map, size=(h, w), mode="bilinear") + feature_map # (shape: (batch_size, channels, h, w)))

        return out

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        c0 = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, 64, h/2, w/2))
        c1 = F.max_pool2d(c0, kernel_size=3, stride=2, padding=1) # (shape: (batch_size, 64, h/4, w/4))
        c2 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) # (shape: (batch_size, 64, h/8, w/8)) # NOTE! Should not actually be here!
        c3 = self.layer2(c2) # (shape: (batch_size, block.expansion*64, h/8, w/8))
        c4 = self.layer3(c3) # (shape: (batch_size, block.expansion*128, h/16, w/16))
        c5 = self.layer4(c4) # (shape: (batch_size, block.expansion*256, h/32, w/32))
        c6 = self.layer5(c5) # (shape: (batch_size, block.expansion*512, h/64, w/64))

        p7 = self.conv6(c6) # (shape: (batch_size, 256, h/128, w/128))
        p8 = self.conv7(F.relu(p7)) # (shape: (batch_size, 256, h/256, w/256))

        p6 = self.lateral_conv5(c6) # (shape: (batch_size, 256, h/64, w/64))

        p5 = self._upsample_and_add(feature_map=self.lateral_conv4(c5),
                                    small_feature_map=p6) # (shape: (batch_size, 256, h/32, w/32))
        p5 = self.smoothing_conv4(p5) # (shape: (batch_size, 256, h/32, w/32))

        p4 = self._upsample_and_add(feature_map=self.lateral_conv3(c4),
                                    small_feature_map=p5) # (shape: (batch_size, 256, h/16, w/16))
        p4 = self.smoothing_conv3(p4) # (shape: (batch_size, 256, h/16, w/16))

        # (p4 has shape: (batch_size, 256, h/16, w/16))
        # (p5 has shape: (batch_size, 256, h/32, w/32))
        # (p6 has shape: (batch_size, 256, h/64, w/64))
        # (p7 has shape: (batch_size, 256, h/128, w/128))
        # (p8 has shape: (batch_size, 256, h/256, w/256))
        return (p4, p5, p6, p7, p8)

def FPN18():
    return FPN(block=BasicBlock, num_blocks=[2, 2, 2, 2])

def FPN34():
    return FPN(block=BasicBlock, num_blocks=[3, 4, 6, 3])

def FPN50():
    return FPN(block=Bottleneck, num_blocks=[3, 4, 6, 3])

def FPN101():
    return FPN(block=Bottleneck, num_blocks=[3, 4, 23, 3])

def FPN152():
    return FPN(block=Bottleneck, num_blocks=[3, 8, 36, 3])
