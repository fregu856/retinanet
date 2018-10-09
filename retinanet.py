# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision.models as models
#
# from fpn_pretrained import FPN18, FPN34, FPN50, FPN101, FPN152
#
# import os
#
# class RetinaNet(nn.Module):
#     def __init__(self, model_id, project_dir):
#         super(RetinaNet, self).__init__()
#
#         self.model_id = model_id
#         self.project_dir = project_dir
#         self.create_model_dirs()
#
#         self.anchors_per_cell = 9 # (num anchors per grid point)
#         self.num_classes = 4 # (background, car, pedestrian, cyclist)
#
#         self.fpn = FPN18() # NOTE! (FPN18, FPN34, FPN50, FPN101 or FPN152)
#
#         self.class_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv5 = nn.Conv2d(256, self.num_classes*self.anchors_per_cell, kernel_size=3, stride=1, padding=1)
#
#         self.regr_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv5 = nn.Conv2d(256, 4*self.anchors_per_cell, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         # (x has shape (batch_size, 3, h, w))
#
#         batch_size = x.size(0)
#
#         p4, p5, p6, p7, p8 = self.fpn(x)
#         # (p4 has shape: (batch_size, 256, h/16, w/16))
#         # (p5 has shape: (batch_size, 256, h/32, w/32))
#         # (p6 has shape: (batch_size, 256, h/64, w/64))
#         # (p7 has shape: (batch_size, 256, h/128, w/128))
#         # (p8 has shape: (batch_size, 256, h/256, w/256))
#
#         outputs_class = []
#         outputs_regr = []
#         for feature_map in [p4, p5, p6, p7, p8]:
#             # (feature_map has shape: (batch_size, 256, h/y, w/y), y in {16, 32, 64, 128, 256})
#
#             out_regr = F.relu(self.regr_conv1(feature_map)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = F.relu(self.regr_conv2(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = F.relu(self.regr_conv3(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = F.relu(self.regr_conv4(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = self.regr_conv5(out_regr) # (shape: (batch_size, 4*self.anchors_per_cell, h/y, w/y))
#             # NOTE! NOTE! or is it this that messes things up?
#             out_regr = out_regr.view(-1, batch_size, 4) # (shape: ((h/y)*(w/y)*self.anchors_per_cell, batch_size, 4))
#             out_regr = out_regr.permute(1, 0, 2) # (shape: (batch_size, (h/y)*(w/y)*self.anchors_per_cell, 4)) (resid_x, resid_y, resid_w, resid_h)
#             outputs_regr.append(out_regr)
#
#             out_class = F.relu(self.class_conv1(feature_map)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = F.relu(self.class_conv2(out_class)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = F.relu(self.class_conv3(out_class)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = F.relu(self.class_conv4(out_class)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = self.class_conv5(out_class) # (shape: (batch_size, self.num_classes*self.anchors_per_cell, h/y, w/y))
#             out_class = out_class.view(-1, batch_size, self.num_classes) # (shape: ((h/y)*(w/y)*self.anchors_per_cell, batch_size, self.num_classes))
#             out_class = out_class.permute(1, 0, 2) # (shape: (batch_size, (h/y)*(w/y)*self.anchors_per_cell, self.num_classes))
#             outputs_class.append(out_class)
#
#         outputs_regr = torch.cat(outputs_regr, 1) # (shape: (batch_size, num_anchors, 4)) (num_anchors: total number of anchors in the image) (resid_x, resid_y, resid_w, resid_h)
#         outputs_class = torch.cat(outputs_class, 1) # (shape: (batch_size, num_anchors, self.num_classes))
#
#         return (outputs_regr, outputs_class)
#
#     def create_model_dirs(self):
#         self.logs_dir = self.project_dir + "/training_logs"
#         self.model_dir = self.logs_dir + "/model_%s" % self.model_id
#         self.checkpoints_dir = self.model_dir + "/checkpoints"
#         if not os.path.exists(self.logs_dir):
#             os.makedirs(self.logs_dir)
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#             os.makedirs(self.checkpoints_dir)
#
# # x = Variable(torch.randn(32, 3, 375, 1242)).cuda()
# # network = RetinaNet("RetinaNet_test", "/staging/frexgus/retinanet")
# # network = network.cuda()
# # out = network(x)
#
# # x = Variable(torch.randn(1, 3, 375, 1242))
# # network = RetinaNet("RetinaNet_test", "/home/fregu856/exjobb/training_logs/retinanet")
# # out = network(x)

# ################################################################################
# # attempt at not using .view() in the computation of the loss:
# ################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torchvision.models as models
#
# from fpn_pretrained import FPN18, FPN34, FPN50, FPN101, FPN152
#
# import os
#
# class RetinaNet(nn.Module):
#     def __init__(self, model_id, project_dir):
#         super(RetinaNet, self).__init__()
#
#         self.model_id = model_id
#         self.project_dir = project_dir
#         self.create_model_dirs()
#
#         self.anchors_per_cell = 9 # (num anchors per grid point)
#         self.num_classes = 4 # (background, car, pedestrian, cyclist)
#
#         self.fpn = FPN18() # NOTE! (FPN18, FPN34, FPN50, FPN101 or FPN152)
#
#         self.class_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.class_conv5 = nn.Conv2d(256, self.num_classes*self.anchors_per_cell, kernel_size=3, stride=1, padding=1)
#
#         self.regr_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.regr_conv5 = nn.Conv2d(256, 4*self.anchors_per_cell, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         # (x has shape (batch_size, 3, h, w))
#
#         batch_size = x.size(0)
#
#         p4, p5, p6, p7, p8 = self.fpn(x)
#         # (p4 has shape: (batch_size, 256, h/16, w/16))
#         # (p5 has shape: (batch_size, 256, h/32, w/32))
#         # (p6 has shape: (batch_size, 256, h/64, w/64))
#         # (p7 has shape: (batch_size, 256, h/128, w/128))
#         # (p8 has shape: (batch_size, 256, h/256, w/256))
#
#         outputs_class = []
#         outputs_regr = []
#         for feature_map in [p4, p5, p6, p7, p8]:
#             # (feature_map has shape: (batch_size, 256, h/y, w/y), y in {16, 32, 64, 128, 256})
#
#             out_regr = F.relu(self.regr_conv1(feature_map)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = F.relu(self.regr_conv2(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = F.relu(self.regr_conv3(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = F.relu(self.regr_conv4(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
#             out_regr = self.regr_conv5(out_regr) # (shape: (batch_size, 4*self.anchors_per_cell, h/y, w/y))
#             # NOTE! NOTE! or is it this that messes things up?
#             out_regr = out_regr.view(-1, batch_size, 4) # (shape: ((h/y)*(w/y)*self.anchors_per_cell, batch_size, 4))
#             out_regr = out_regr.permute(1, 0, 2) # (shape: (batch_size, (h/y)*(w/y)*self.anchors_per_cell, 4)) (resid_x, resid_y, resid_w, resid_h)
#             outputs_regr.append(out_regr)
#
#             out_class = F.relu(self.class_conv1(feature_map)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = F.relu(self.class_conv2(out_class)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = F.relu(self.class_conv3(out_class)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = F.relu(self.class_conv4(out_class)) # (shape: (batch_size, 256, h/y, w/y))
#             out_class = self.class_conv5(out_class) # (shape: (batch_size, self.num_classes*self.anchors_per_cell, h/y, w/y))
#             out_class = out_class.view(-1, batch_size, self.num_classes) # (shape: ((h/y)*(w/y)*self.anchors_per_cell, batch_size, self.num_classes))
#             out_class = out_class.permute(1, 2, 0) # (shape: (batch_size, self.num_classes, (h/y)*(w/y)*self.anchors_per_cell))
#             outputs_class.append(out_class)
#
#         outputs_regr = torch.cat(outputs_regr, 1) # (shape: (batch_size, num_anchors, 4)) (num_anchors: total number of anchors in the image) (resid_x, resid_y, resid_w, resid_h)
#         outputs_class = torch.cat(outputs_class, 2) # (shape: (batch_size, self.num_classes, num_anchors))
#
#         return (outputs_regr, outputs_class)
#
#     def create_model_dirs(self):
#         self.logs_dir = self.project_dir + "/training_logs"
#         self.model_dir = self.logs_dir + "/model_%s" % self.model_id
#         self.checkpoints_dir = self.model_dir + "/checkpoints"
#         if not os.path.exists(self.logs_dir):
#             os.makedirs(self.logs_dir)
#         if not os.path.exists(self.model_dir):
#             os.makedirs(self.model_dir)
#             os.makedirs(self.checkpoints_dir)
#
# # x = Variable(torch.randn(32, 3, 375, 1242)).cuda()
# # network = RetinaNet("RetinaNet_test", "/staging/frexgus/retinanet")
# # network = network.cuda()
# # out = network(x)
#
# # x = Variable(torch.randn(1, 3, 375, 1242))
# # network = RetinaNet("RetinaNet_test", "/home/fregu856/exjobb/training_logs/retinanet")
# # out = network(x)

################################################################################
# second attempt at fix (contigous):
################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from fpn_pretrained import FPN18, FPN34, FPN50, FPN101, FPN152

import os

class RetinaNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(RetinaNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.anchors_per_cell = 9 # (num anchors per grid point)
        self.num_classes = 4 # (background, car, pedestrian, cyclist)

        self.fpn = FPN18() # NOTE! (FPN18, FPN34, FPN50, FPN101 or FPN152)

        self.class_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.class_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.class_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.class_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.class_conv5 = nn.Conv2d(256, self.num_classes*self.anchors_per_cell, kernel_size=3, stride=1, padding=1)

        self.regr_conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.regr_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.regr_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.regr_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.regr_conv5 = nn.Conv2d(256, 4*self.anchors_per_cell, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        batch_size = x.size(0)

        p4, p5, p6, p7, p8 = self.fpn(x)
        # (p4 has shape: (batch_size, 256, h/16, w/16))
        # (p5 has shape: (batch_size, 256, h/32, w/32))
        # (p6 has shape: (batch_size, 256, h/64, w/64))
        # (p7 has shape: (batch_size, 256, h/128, w/128))
        # (p8 has shape: (batch_size, 256, h/256, w/256))

        outputs_class = []
        outputs_regr = []
        for feature_map in [p4, p5, p6, p7, p8]:
            # (feature_map has shape: (batch_size, 256, h/y, w/y), y in {16, 32, 64, 128, 256})

            out_regr = F.relu(self.regr_conv1(feature_map)) # (shape: (batch_size, 256, h/y, w/y))
            out_regr = F.relu(self.regr_conv2(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
            out_regr = F.relu(self.regr_conv3(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
            out_regr = F.relu(self.regr_conv4(out_regr)) # (shape: (batch_size, 256, h/y, w/y))
            out_regr = self.regr_conv5(out_regr) # (shape: (batch_size, 4*self.anchors_per_cell, h/y, w/y))
            # # NOTE! NOTE! or is it this that messes things up?
            # out_regr = out_regr.view(-1, batch_size, 4) # (shape: ((h/y)*(w/y)*self.anchors_per_cell, batch_size, 4))
            # out_regr = out_regr.permute(1, 0, 2) # (shape: (batch_size, (h/y)*(w/y)*self.anchors_per_cell, 4)) (resid_x, resid_y, resid_w, resid_h)
            out_regr = out_regr.permute(0, 2, 3, 1) # (shape: (batch_size, h/y, w/y, 4*self.anchors_per_cell))
            out_regr = out_regr.contiguous().view(out_regr.size(0), -1, 4) # (shape: (batch_size, (h/y)*(w/y)*self.anchors_per_cell, 4))
            outputs_regr.append(out_regr)

            out_class = F.relu(self.class_conv1(feature_map)) # (shape: (batch_size, 256, h/y, w/y))
            out_class = F.relu(self.class_conv2(out_class)) # (shape: (batch_size, 256, h/y, w/y))
            out_class = F.relu(self.class_conv3(out_class)) # (shape: (batch_size, 256, h/y, w/y))
            out_class = F.relu(self.class_conv4(out_class)) # (shape: (batch_size, 256, h/y, w/y))
            out_class = self.class_conv5(out_class) # (shape: (batch_size, self.num_classes*self.anchors_per_cell, h/y, w/y))
            # out_class = out_class.view(-1, batch_size, self.num_classes) # (shape: ((h/y)*(w/y)*self.anchors_per_cell, batch_size, self.num_classes))
            # out_class = out_class.permute(1, 2, 0) # (shape: (batch_size, self.num_classes, (h/y)*(w/y)*self.anchors_per_cell))
            out_class = out_class.permute(0, 2, 3, 1) # (shape: (batch_size, h/y, w/y, self.num_classes*self.anchors_per_cell))
            out_class = out_class.contiguous().view(out_class.size(0), self.num_classes, -1) # (shape: (batch_size, self.num_classes, (h/y)*(w/y)*self.anchors_per_cell))
            outputs_class.append(out_class)

        outputs_regr = torch.cat(outputs_regr, 1) # (shape: (batch_size, num_anchors, 4)) (num_anchors: total number of anchors in the image) (resid_x, resid_y, resid_w, resid_h)
        outputs_class = torch.cat(outputs_class, 2) # (shape: (batch_size, self.num_classes, num_anchors))

        return (outputs_regr, outputs_class)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

# x = Variable(torch.randn(32, 3, 375, 1242)).cuda()
# network = RetinaNet("RetinaNet_test", "/staging/frexgus/retinanet")
# network = network.cuda()
# out = network(x)

# x = Variable(torch.randn(1, 3, 375, 1242))
# network = RetinaNet("RetinaNet_test", "/home/fregu856/exjobb/training_logs/retinanet")
# out = network(x)
