import torch
import torch.nn as nn
from torch.autograd import Variable

import math

def gradClamp(parameters, clip):
    for p in parameters:
        p.grad.data = p.grad.data.clamp_(max=clip)

def onehot_embed(labels, num_classes):
    # (labels is a Variable of dtype LongTensor and shape: (n, ))

    labels = labels.data.cpu()

    onehot_labels = torch.eye(num_classes) # (shape: (num_classes, num_classes))
    onehot_labels = onehot_labels[labels] # (shape: (n, num_classes))
    onehot_labels = Variable(onehot_labels).cuda()

    return onehot_labels

# def init_weights(network):
#     for m in network.modules():
#         if isinstance(m, nn.Conv2d):
#             m.weight.data.normal_(0.0, 0.01)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1.0)
#             m.bias.data.fill_(0.0)

def init_weights(network):
    network.class_conv1.weight.data.normal_(0.0, 0.01)
    network.class_conv1.bias.data.fill_(0.0)
    network.class_conv2.weight.data.normal_(0.0, 0.01)
    network.class_conv2.bias.data.fill_(0.0)
    network.class_conv3.weight.data.normal_(0.0, 0.01)
    network.class_conv3.bias.data.fill_(0.0)
    network.class_conv4.weight.data.normal_(0.0, 0.01)
    network.class_conv4.bias.data.fill_(0.0)
    network.class_conv5.weight.data.normal_(0.0, 0.01)
    network.class_conv5.bias.data.fill_(0.0)

    network.regr_conv1.weight.data.normal_(0.0, 0.01)
    network.regr_conv1.bias.data.fill_(0.0)
    network.regr_conv2.weight.data.normal_(0.0, 0.01)
    network.regr_conv2.bias.data.fill_(0.0)
    network.regr_conv3.weight.data.normal_(0.0, 0.01)
    network.regr_conv3.bias.data.fill_(0.0)
    network.regr_conv4.weight.data.normal_(0.0, 0.01)
    network.regr_conv4.bias.data.fill_(0.0)
    network.regr_conv5.weight.data.normal_(0.0, 0.01)
    network.regr_conv5.bias.data.fill_(0.0)

    network.fpn.conv6.weight.data.normal_(0.0, 0.01)
    network.fpn.conv6.bias.data.fill_(0.0)
    network.fpn.conv7.weight.data.normal_(0.0, 0.01)
    network.fpn.conv7.bias.data.fill_(0.0)

    network.fpn.lateral_conv5.weight.data.normal_(0.0, 0.01)
    network.fpn.lateral_conv5.bias.data.fill_(0.0)
    network.fpn.lateral_conv4.weight.data.normal_(0.0, 0.01)
    network.fpn.lateral_conv4.bias.data.fill_(0.0)
    network.fpn.lateral_conv3.weight.data.normal_(0.0, 0.01)
    network.fpn.lateral_conv3.bias.data.fill_(0.0)

    network.fpn.smoothing_conv4.weight.data.normal_(0.0, 0.01)
    network.fpn.smoothing_conv4.bias.data.fill_(0.0)
    network.fpn.smoothing_conv3.weight.data.normal_(0.0, 0.01)
    network.fpn.smoothing_conv3.bias.data.fill_(0.0)

def add_weight_decay(net, l2_value, skip_list=()):
    # https://raberrytv.wordpress.com/2017/10/29/pytorch-weight-decay-made-easy/

    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]
