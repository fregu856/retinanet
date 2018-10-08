import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class FPN_Bottleneck(nn.Module):
    def __init__(self, num_layers):
        super(FPN_Bottleneck, self).__init__()

        if num_layers == 50:
            resnet = models.resnet50()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/staging/frexgus/frustum_pointnet/resnet50-19c8e357.pth"))
            # remove fully connected layer and avg pool:
            self.resnet_layers = nn.ModuleList(list(resnet.children())[:-2])
            print ("pretrained resnet, 50")
        # elif num_layers == 101:
        #     resnet = models.resnet101()
        #     # load pretrained model:
        #     resnet.load_state_dict(torch.load("/staging/frexgus/frustum_pointnet/resnet34-333f7ec4.pth"))
        #     # remove fully connected layer and avg pool:
        #     self.resnet_layers = nn.ModuleList(list(resnet.children())[:-2])
        #     print ("pretrained resnet, 101")
        # elif num_layers == 152:
        #     resnet = models.resnet152()
        #     # load pretrained model:
        #     resnet.load_state_dict(torch.load("/staging/frexgus/frustum_pointnet/resnet34-333f7ec4.pth"))
        #     # remove fully connected layer and avg pool:
        #     self.resnet_layers = nn.ModuleList(list(resnet.children())[:-2])
        #     print ("pretrained resnet, 152")
        else:
            raise Exception("num_layers must be in {50, 101, 152}!")

        self.conv6 = nn.Conv2d(4*512, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.lateral_conv5 = nn.Conv2d(4*512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv4 = nn.Conv2d(4*256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(4*128, 256, kernel_size=1, stride=1, padding=0)

        self.smoothing_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothing_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_and_add(self, feature_map, small_feature_map):
        # (feature_map has shape (batch_size, channels, h, w))
        # (small_feature_map has shape (batch_size, channels, h/2, w/2)) (integer division)

        _, _, h, w = feature_map.size()

        out = F.upsample(small_feature_map, size=(h, w), mode="bilinear") + feature_map # (shape: (batch_size, channels, h, w)))

        return out

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) # (shape: (batch_size, 3, h/2, w/2))

        # pass x through the pretrained ResNet and collect feature maps:
        c = []
        for layer in self.resnet_layers:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                c.append(x)

        # NOTE! all spatial dimensons below should actually be divided by 2 (because of the initial max pool)

        c2 = c[0] # (shape: (batch_size, 4*64, h/4, w/4))
        c3 = c[1] # (shape: (batch_size, 4*128, h/8, w/8))
        c4 = c[2] # (shape: (batch_size, 4*256, h/16, w/16))
        c5 = c[3] # (shape: (batch_size, 4*512, h/32, w/32))

        p6 = self.conv6(c5) # (shape: (batch_size, 256, h/64, w/64))
        p7 = self.conv7(F.relu(p6)) # (shape: (batch_size, 256, h/128, w/128))

        p5 = self.lateral_conv5(c5) # (shape: (batch_size, 256, h/32, w/32))

        p4 = self._upsample_and_add(feature_map=self.lateral_conv4(c4),
                                    small_feature_map=p5) # (shape: (batch_size, 256, h/16, w/16))
        p4 = self.smoothing_conv4(p4) # (shape: (batch_size, 256, h/16, w/16))

        p3 = self._upsample_and_add(feature_map=self.lateral_conv3(c3),
                                    small_feature_map=p4) # (shape: (batch_size, 256, h/8, w/8))
        p3 = self.smoothing_conv3(p3) # (shape: (batch_size, 256, h/8, w/8))

        # (p3 has shape: (batch_size, 256, h/8, w/8))
        # (p4 has shape: (batch_size, 256, h/16, w/16))
        # (p5 has shape: (batch_size, 256, h/32, w/32))
        # (p6 has shape: (batch_size, 256, h/64, w/64))
        # (p7 has shape: (batch_size, 256, h/128, w/128))
        return (p3, p4, p5, p6, p7)

class FPN_BasicBlock(nn.Module):
    def __init__(self, num_layers):
        super(FPN_BasicBlock, self).__init__()

        if num_layers == 18:
            resnet = models.resnet18()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/staging/frexgus/frustum_pointnet/resnet18-5c106cde.pth"))
            # remove fully connected layer and avg pool:
            self.resnet_layers = nn.ModuleList(list(resnet.children())[:-2])
            print ("pretrained resnet, 18")
        elif num_layers == 34:
            resnet = models.resnet34()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/staging/frexgus/frustum_pointnet/resnet34-333f7ec4.pth"))
            # remove fully connected layer and avg pool:
            self.resnet_layers = nn.ModuleList(list(resnet.children())[:-2])
            print ("pretrained resnet, 34")
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.conv6 = nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.lateral_conv5 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.smoothing_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smoothing_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _upsample_and_add(self, feature_map, small_feature_map):
        # (feature_map has shape (batch_size, channels, h, w))
        # (small_feature_map has shape (batch_size, channels, h/2, w/2)) (integer division)

        _, _, h, w = feature_map.size()

        out = F.upsample(small_feature_map, size=(h, w), mode="bilinear") + feature_map # (shape: (batch_size, channels, h, w)))

        return out

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1) # (shape: (batch_size, 3, h/2, w/2))

        # pass x through the pretrained ResNet and collect feature maps:
        c = []
        for layer in self.resnet_layers:
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                c.append(x)

        # NOTE! all spatial dimensons below should actually be divided by 2 (because of the initial max pool)

        c2 = c[0] # (shape: (batch_size, 64, h/4, w/4))
        c3 = c[1] # (shape: (batch_size, 128, h/8, w/8))
        c4 = c[2] # (shape: (batch_size, 256, h/16, w/16))
        c5 = c[3] # (shape: (batch_size, 512, h/32, w/32))

        p6 = self.conv6(c5) # (shape: (batch_size, 256, h/64, w/64))
        p7 = self.conv7(F.relu(p6)) # (shape: (batch_size, 256, h/128, w/128))

        p5 = self.lateral_conv5(c5) # (shape: (batch_size, 256, h/32, w/32))

        p4 = self._upsample_and_add(feature_map=self.lateral_conv4(c4),
                                    small_feature_map=p5) # (shape: (batch_size, 256, h/16, w/16))
        p4 = self.smoothing_conv4(p4) # (shape: (batch_size, 256, h/16, w/16))

        p3 = self._upsample_and_add(feature_map=self.lateral_conv3(c3),
                                    small_feature_map=p4) # (shape: (batch_size, 256, h/8, w/8))
        p3 = self.smoothing_conv3(p3) # (shape: (batch_size, 256, h/8, w/8))

        # (p3 has shape: (batch_size, 256, h/8, w/8))
        # (p4 has shape: (batch_size, 256, h/16, w/16))
        # (p5 has shape: (batch_size, 256, h/32, w/32))
        # (p6 has shape: (batch_size, 256, h/64, w/64))
        # (p7 has shape: (batch_size, 256, h/128, w/128))
        return (p3, p4, p5, p6, p7)

def FPN18():
    return FPN_BasicBlock(num_layers=18)

def FPN34():
    return FPN_BasicBlock(num_layers=34)

def FPN50():
    return FPN_Bottleneck(num_layers=50)

def FPN101():
    return FPN_Bottleneck(num_layers=101)

def FPN152():
    return FPN_Bottleneck(num_layers=152)

# x = Variable(torch.randn(1, 3, 512, 512))
# network = FPN_BasicBlock(num_layers=34)
# out = network(x)

# x = Variable(torch.randn(1, 3, 512, 512))
# network = FPN_Bottleneck(num_layers=50)
# out = network(x)
