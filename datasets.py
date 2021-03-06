from kittiloader import LabelLoader2D3D, LabelLoader2D3D_sequence # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

import sys

sys.path.append("/root/retinanet/data_aug")
sys.path.append("/home/fregu856/retinanet/data_aug")
from data_aug import RandomHorizontalFlip, RandomHSV, RandomScale, RandomTranslate, Resize

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import pickle
import numpy as np
import cv2
import math
import os

class_string_to_label = {"Car": 1,
                         "Pedestrian": 2,
                         "Cyclist": 3} # (background: 0)

################################################################################
# debug visualization helper functions START
################################################################################
def create2Dbbox_poly(bbox2D):
    u_min = bbox2D[0] # (left)
    u_max = bbox2D[1] # (rigth)
    v_min = bbox2D[2] # (top)
    v_max = bbox2D[3] # (bottom)

    poly = {}
    poly['poly'] = np.array([[u_min, v_min], [u_max, v_min],
                             [u_max, v_max], [u_min, v_max]], dtype='int32')

    return poly

def draw_2d_polys_no_text(img, polys):
    img = np.copy(img)
    for poly in polys:
        if 'color' in poly:
            bg = poly['color']
        else:
            bg = np.array([0, 255, 0], dtype='float64')

        cv2.polylines(img, np.int32([poly['poly']]), True, bg, lineType=cv2.LINE_AA, thickness=2)

    return img
################################################################################
# debug visualization helper functions END
################################################################################

def bboxes_xxyyc_2_xyxyc(bboxes_xxyyc):
    # (bboxes_xxyyc is an array of shape (num_bboxes, 5), (x_min, x_max, y_min, y_max, class_label))

    bboxes_xyxyc = np.zeros(bboxes_xxyyc.shape, dtype=bboxes_xxyyc.dtype)

    bboxes_xyxyc[:, 0] = bboxes_xxyyc[:, 0]
    bboxes_xyxyc[:, 1] = bboxes_xxyyc[:, 2]
    bboxes_xyxyc[:, 2] = bboxes_xxyyc[:, 1]
    bboxes_xyxyc[:, 3] = bboxes_xxyyc[:, 3]
    bboxes_xyxyc[:, 4] = bboxes_xxyyc[:, 4]

    # (bboxes_xyxyc is an array of shape (num_bboxes, 5), (x_min, y_min, x_max, y_max, class_label))
    return bboxes_xyxyc

def bboxes_xyxyc_2_xxyyc(bboxes_xyxyc):
    # (bboxes_xyxyc is an array of shape (num_bboxes, 5), (x_min, y_min, x_max, y_max, class_label))

    bboxes_xxyyc = np.zeros(bboxes_xyxyc.shape, dtype=bboxes_xyxyc.dtype)

    bboxes_xxyyc[:, 0] = bboxes_xyxyc[:, 0]
    bboxes_xxyyc[:, 1] = bboxes_xyxyc[:, 2]
    bboxes_xxyyc[:, 2] = bboxes_xyxyc[:, 1]
    bboxes_xxyyc[:, 3] = bboxes_xyxyc[:, 3]
    bboxes_xxyyc[:, 4] = bboxes_xyxyc[:, 4]

    # (bboxes_xxyyc is an array of shape (num_bboxes, 5), (x_min, x_max, y_min, y_max, class_label))
    return bboxes_xxyyc

class BboxEncoder:
    # NOTE! based off of https://github.com/kuangliu/pytorch-retinanet/blob/master/encoder.py and https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py

    def __init__(self, img_h, img_w):
        self.anchor_areas = [32.0*32.0, 64.0*64.0, 128.0*128.0, 256.0*256.0, 512.0*512.0] # (areas for p4, p5, p6, p7, p8)
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scale_ratios = [1.0, pow(2, 1.0/3.0), pow(2, 2.0/3.0)]

        self.nms_thresh = 0.5
        self.conf_thresh = 0.25

        self.img_h = img_h
        self.img_w = img_w
        self.img_size = torch.Tensor([self.img_w, self.img_h])

        self.anchors_per_cell = 9 # (3 aspect ratios * 3 scale ratios)
        self.num_feature_maps = len(self.anchor_areas) # (p3, p4, p5, p6, p7)

        # (p3 has shape: (batch_size, 256, h/8, w/8))
        # (p4 has shape: (batch_size, 256, h/16, w/16))
        # (p5 has shape: (batch_size, 256, h/32, w/32))
        # (p6 has shape: (batch_size, 256, h/64, w/64))
        # (p7 has shape: (batch_size, 256, h/128, w/128))
        self.feature_map_sizes = [(self.img_size/pow(2.0, i+3)).ceil() for i in range(self.num_feature_maps)]

        self.anchor_sizes = self._get_anchor_sizes() # (Tensor of shape: (num_feature_maps, anchors_per_cell, 2)) (w, h)

        self.anchor_bboxes = self._get_anchor_bboxes() # (Tensor of shape: (num_anchors, 4), (x, y, w, h), where num_anchors == fm1_h*fm1_w*anchors_per_cell + ... + fmN_h*fmN_w*anchors_per_cell)

        self.num_anchors = self.anchor_bboxes.size(0) # (total number of anchor bboxes, num_anchors == fm1_h*fm1_w*anchors_per_cell + ... + fmN_h*fmN_w*anchors_per_cell)

        print (self.num_anchors)

    def _get_anchor_sizes(self):
        anchor_sizes = []
        for area in self.anchor_areas:
            for aspect_ratio in self.aspect_ratios:
                h = math.sqrt(area/aspect_ratio)
                w = aspect_ratio*h
                for scale_ratio in self.scale_ratios:
                    anchor_h = scale_ratio*h
                    anchor_w = scale_ratio*w
                    anchor_sizes.append([anchor_w, anchor_h])
        anchor_sizes = torch.Tensor(anchor_sizes).view(self.num_feature_maps, self.anchors_per_cell, 2)

        return anchor_sizes

    def _mesh_grid(self, x, y):
        # _mesh_grid(x, y) is a Tensor of shape (x*y, 2)
        # _mesh_grid(3, 2):
        # 0  0
        # 1  0
        # 2  0
        # 0  1
        # 1  1
        # 2  1

        x_range = torch.arange(0, x) # (Tensor of shape (x, ): (0, 1, 2,..., x-1))
        y_range = torch.arange(0, y) # (Tensor of shape (y, ): (0, 1, 2,..., y-1))

        xx = x_range.repeat(y).view(-1, 1) # (Tensor of shape: (x*y, 1). x == 3, y == 2: xx == (0, 1, 2, 0, 1, 2))
        yy = y_range.view(-1, 1).repeat(1, x).view(-1, 1) # (Tensor of shape: (x*y, 1). x == 3, y ==2: yy == (0, 0, 0, 1, 1, 1))

        mesh_grid = torch.cat([xx, yy], 1) # (Tensor of shape: (x*y, 2). mesh_grid[:, 0] == xx, mesh_grid[:, 1] == yy)

        mesh_grid = mesh_grid.type(torch.FloatTensor)

        return mesh_grid

    def _get_anchor_bboxes(self):
        anchor_bboxes = []
        for i in range(self.num_feature_maps):
            fm_size = self.feature_map_sizes[i]

            grid_cell_size = self.img_size/fm_size # (Tensor of shape (2, ), (cell_w, cell_h))

            fm_w = int(fm_size[0])
            fm_h = int(fm_size[1])

            grid_cell_centers = self._mesh_grid(fm_w, fm_h) + 0.5
            # (Tensor of shape: (fm_w*fm_h, 2). fm_w == 3, fm_h == 2: grid_cell_centers ==
            # 0.5 0.5
            # 1.5 0.5
            # 2.5 0.5
            # 0.5 1.5
            # 1.5 1.5
            # 2.5 1.5)

            grid_cell_pixel_centers = grid_cell_size*grid_cell_centers
            # (Tensor of shape: (fm_w*fm_h, 2). fm_w == 3, fm_h == 2, grid_cell_size == (10, 10):
            # grid_cell_centers ==
            # 5 5
            # 15 5
            # 25 5
            # 5 15
            # 15 15
            # 25 15)

            anchor_bboxes_x_y = grid_cell_pixel_centers.view(fm_h, fm_w, 1, 2) # (Tensor of shape: (fm_h, fm_w, 1, 2))
            anchor_bboxes_x_y = anchor_bboxes_x_y.expand(fm_h, fm_w, self.anchors_per_cell, 2) # (Tensor of shape: (fm_h, fm_w, anchors_per_cell, 2)
            # (fm_w == 2, fm_h == 2, grid_cell_size == (10, 10), anchors_per_cell == 4:
            # anchor_bboxes_x_y[0, 0, :, :] ==
            #    5   5
            #    5   5
            #    5   5
            #    5   5
            # anchor_bboxes_x_y[0, 1, :, :] ==
            #   15   5
            #   15   5
            #   15   5
            #   15   5
            # anchor_bboxes_x_y[0, 2, :, :] ==
            #   25   5
            #   25   5
            #   25   5
            #   25   5
            # anchor_bboxes_x_y[1, 0, :, :] ==
            #    5  15
            #    5  15
            #    5  15
            #    5  15
            # anchor_bboxes_x_y[1, 1, :, :] ==
            #   15  15
            #   15  15
            #   15  15
            #   15  15
            # anchor_bboxes_x_y[1, 2, :, :] ==
            #   25  15
            #   25  15
            #   25  15
            #   25  15

            # (self.anchor_sizes is a Tensor of shape: (num_feature_maps, anchors_per_cell, 2)) (w, h)

            anchor_bboxes_w_h = self.anchor_sizes[i].view(1, 1, self.anchors_per_cell, 2) # (Tensor of shape: (1, 1, anchors_per_cell, 2))
            anchor_bboxes_w_h = anchor_bboxes_w_h.expand(fm_h, fm_w, self.anchors_per_cell, 2) # (Tensor of shape: (fm_h, fm_w, anchors_per_cell, 2))
            # (anchor_bboxes_w_h[i, j, :, :] are the same for all i, j)

            # (anchor_bboxes_x_y and anchor_bboxes_w_h are both Tensors of shape: (fm_h, fm_w, anchors_per_cell, 2))

            anchor_bboxes_x_y_w_h = torch.cat([anchor_bboxes_x_y, anchor_bboxes_w_h], 3) # (Tensor of shape: (fm_h, fm_w, anchors_per_cell, 4), (x, y, w, h))

            anchor_bboxes_x_y_w_h = anchor_bboxes_x_y_w_h.view(-1, 4) # (Tensor of shape: (fm_h*fm_w*anchors_per_cell, 4), (x, y, w, h))
            anchor_bboxes.append(anchor_bboxes_x_y_w_h)

        anchor_bboxes = torch.cat(anchor_bboxes, 0) # (Tensor of shape: (num_anchors, 4), (x, y, w, h), where num_anchors == fm1_h*fm1_w*anchors_per_cell + ... + fmN_h*fmN_w*anchors_per_cell)

        return anchor_bboxes

    def _xxyy_2_xywh(self, bboxes):
        # (bboxes is a Tensor of shape (num_bboxes, 4), (x_min, x_max, y_min, y_max))

        x_min = bboxes[:, 0]
        x_max = bboxes[:, 1]
        y_min = bboxes[:, 2]
        y_max = bboxes[:, 3]

        w = x_max - x_min
        h = y_max - y_min

        x = x_min + w/2.0
        y = y_min + h/2.0

        bboxes = torch.cat([x.view(-1, 1), y.view(-1, 1), w.view(-1, 1), h.view(-1, 1)], 1) # (shape: (num_bboxes, 4), (x, y, w, h))

        return bboxes

    def _xywh_2_xxyy(self, bboxes):
        # (bboxes is a Tensor of shape (num_bboxes, 4), (x, y, w, h))

        x = bboxes[:, 0]
        y = bboxes[:, 1]
        w = bboxes[:, 2]
        h = bboxes[:, 3]

        x_min = x - w/2.0
        x_max = x + w/2.0

        y_min = y - h/2.0
        y_max = y + h/2.0

        bboxes = torch.cat([x_min.view(-1, 1), x_max.view(-1, 1), y_min.view(-1, 1), y_max.view(-1, 1)], 1) # (shape: (num_bboxes, 4), (x_min, x_max, y_min, y_max))

        return bboxes

    def _bboxes_ious(self, anchor_bboxes, gt_bboxes):
        # (anchor_bboxes is a Tensor of shape (num_anchors, 4), (x, y, w, h))
        # (gt_bboxes is a Tensor of shape (num_gt_objects, 4), (x, y, w, h))

        intersect_xmax = np.minimum(anchor_bboxes[:, None, 0] + 0.5*anchor_bboxes[:, None, 2],
                                    gt_bboxes[:, 0] + 0.5*gt_bboxes[:, 2]) # (shape (num_anchors, num_gt_objects))
        intersect_xmin = np.maximum(anchor_bboxes[:, None, 0] - 0.5*anchor_bboxes[:, None, 2],
                                    gt_bboxes[:, 0] - 0.5*gt_bboxes[:, 2]) # (shape (num_anchors, num_gt_objects))
        intersect_ymax = np.minimum(anchor_bboxes[:, None, 1] + 0.5*anchor_bboxes[:, None, 3],
                                    gt_bboxes[:, 1] + 0.5*gt_bboxes[:, 3]) # (shape (num_anchors, num_gt_objects))
        intersect_ymin = np.maximum(anchor_bboxes[:, None, 1] - 0.5*anchor_bboxes[:, None, 3],
                                    gt_bboxes[:, 1] - 0.5*gt_bboxes[:, 3]) # (shape (num_anchors, num_gt_objects))

        zeros = torch.zeros(intersect_xmin.size()) # (shape (num_anchors, num_gt_object))
        intersect_w = torch.max(zeros, intersect_xmax - intersect_xmin) # (shape (num_anchors, num_gt_object))
        intersect_h = torch.max(zeros, intersect_ymax - intersect_ymin) # (shape (num_anchors, num_gt_object))
        intersection_area = intersect_w*intersect_h # (shape (num_anchors, num_gt_object))

        union_area = anchor_bboxes[:, None, 2]*anchor_bboxes[:, None, 3] + gt_bboxes[:, 2]*gt_bboxes[:, 3] - intersection_area # (shape (num_anchors, num_gt_object))

        ious = intersection_area/union_area # (shape (num_anchors, num_gt_object))
        # (ious[i, j]: the IoU of anchor bbox i with gt bbox j)

        return ious

    def _batch_ious(self, boxes, box):
        # (boxes is a Tensor of shape (num_boxes, 4), (x, y, w, h))
        # (box is a Tensor of shape (4, ), (x, y, w, h))

        intersect_xmax = np.minimum(boxes[:, 0] + 0.5*boxes[:, 2], box[0] + 0.5*box[2]) # (shape (num_boxes, ))
        intersect_xmin = np.maximum(boxes[:, 0] - 0.5*boxes[:, 2], box[0] - 0.5*box[2]) # (shape (num_boxes, ))
        intersect_ymax = np.minimum(boxes[:, 1] + 0.5*boxes[:, 3], box[1] + 0.5*box[3]) # (shape (num_boxes, ))
        intersect_ymin = np.maximum(boxes[:, 1] - 0.5*boxes[:, 3], box[1] - 0.5*box[3]) # (shape (num_boxes, ))

        zeros = torch.zeros(intersect_xmin.size()) # (shape (num_boxes, ))
        intersect_w = torch.max(zeros, intersect_xmax - intersect_xmin) # (shape (num_boxes, ))
        intersect_h = torch.max(zeros, intersect_ymax - intersect_ymin) # (shape (num_boxes, ))
        intersection_area = intersect_w*intersect_h # (shape (num_boxes, ))

        union_area = boxes[:, 2]*boxes[:, 3] + box[2]*box[3] - intersection_area # (shape (num_boxes, ))

        ious = intersection_area/union_area # (shape (num_boxes, ))
        # (ious[i]: the IoU of box with boxes[i])

        return ious

    def _bbox_nms(self, bboxes, scores):
        # NOTE! based off of function from github.com/BichenWuUCB/squeezeDet

        # (bboxes has shape (num_bboxes, 4), (x, y, w, h))
        # (scores has shape (num_bboxes, ))

        num_bboxes = bboxes.size(0)

        # get indices in descending order according to score:
        _, order = scores.sort(0, descending=True) # (shape: (num_bboxes, ))

        keep = torch.ones((order.size(0), )).type(torch.LongTensor)
        for i in range(num_bboxes-1):
            ious = self._batch_ious(bboxes[order[i+1:]], bboxes[order[i]])
            for j, iou in enumerate(ious):
                if iou > self.nms_thresh:
                    keep[order[j+i+1]] = 0

        keep_inds = keep.nonzero() # (shape: (num_bboxes_after_nms, 1))
        keep_inds = keep_inds.squeeze() # (shape: (num_bboxes_after_nms, ))

        return keep_inds

    def encode(self, gt_bboxes, gt_classes):
        # (self.anchor_bboxes is a Tensor of shape: (num_anchors, 4), (x, y, w, h))
        # (gt_bboxes is a Tensor of shape (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        # (gt_classes is a Tensor of shape (num_gt_objects, ))

        gt_bboxes = self._xxyy_2_xywh(gt_bboxes) # (shape: (num_gt_objects, 4), (x ,y, w ,h))

        # compute the IoU of each anchor bbox with each gt bbox:
        ious = self._bboxes_ious(self.anchor_bboxes, gt_bboxes) # (shape (num_anchors, num_gt_object))
        # (ious[i, j]: the IoU of anchor bbox i with gt bbox j)

        # for each anchor bbox, get the maximum IoU and the index of the corresponding gt bbox:
        max_ious, max_inds = ious.max(1) # (both has shape: (num_anchors, ))

        # for each anchor bbox, get the gt bbox corresponding to the maximum IoU:
        assigned_gt_bboxes = gt_bboxes[max_inds] # (shape: (num_anchors, 4), (x, y, w, h))
        # (assigned_gt_bboxes[i]: the gt bbox which is closest (in terms of IoU) to anchor bbox i)

        # target_x = (gt_x - anchor_x)/anchor_w:
        target_x = (assigned_gt_bboxes[:, 0] - self.anchor_bboxes[:, 0])/self.anchor_bboxes[:, 2] # (shape (num_anchors, ))
        target_x = target_x.view(-1, 1) # (shape (num_anchors, 1))
        # target_y = (gt_y - anchor_y)/anchor_h:
        target_y = (assigned_gt_bboxes[:, 1] - self.anchor_bboxes[:, 1])/self.anchor_bboxes[:, 3]
        target_y = target_y.view(-1, 1)
        # target_w = log(gt_w/anchor_w):
        target_w = torch.log(assigned_gt_bboxes[:, 2]/self.anchor_bboxes[:, 2])
        target_w = target_w.view(-1, 1)
        # target_h = log(gt_h/anchor_h):
        target_h = torch.log(assigned_gt_bboxes[:, 3]/self.anchor_bboxes[:, 3])
        target_h = target_h.view(-1, 1)

        labels_regr = torch.cat([target_x, target_y, target_w, target_h], 1) # (shape (num_anchors, 4), (x, y, w, h))

        # for each anchor bbox, get the class label of the gt bbox corresponding to the max IoU:
        assigned_gt_classes = gt_classes[max_inds] # (shape (num_anchors, ))

        # assign all anchor bboxes with maximum IoU < 0.4 to background (0):
        assigned_gt_classes[max_ious < 0.4] = 0

        # assign all anchors which should be ignored during training to -1:
        ignore_inds = (max_ious >= 0.4) & (max_ious < 0.5)
        assigned_gt_classes[ignore_inds] = -1

        labels_class = assigned_gt_classes # (shape (num_anchors, ), entries are in {-1, 0, 1,..., num_classes-1})

        # (labels_regr has shape (num_anchors, 4), (x, y, w, h))
        # (labels_class has shape (num_anchors, ))
        return (labels_regr, labels_class)

    def decode(self, outputs_regr, outputs_class):
        # (outputs_regr has shape (num_anchors, 4), (x, y, w, h))
        # (outputs_class has shape (num_anchors, num_classes))
        # (self.anchor_bboxes has shape (num_anchors, 4), (x, y, w, h))

        # for each anchor bbox, get the pred class label and the corresponding pred score:
        pred_scores = F.softmax(Variable(outputs_class), dim=1).data # (shape (num_anchors, num_classes))
        pred_max_scores, pred_class_labels = torch.max(pred_scores, 1) # (both have shape (num_anchors, ))

        # get the indices of all pred non-background bboxes:
        keep_inds = pred_class_labels != 0 # (shape (num_anchors, ), entries in {0, 1})
        keep_inds = keep_inds.nonzero() # (shape (num_foreground_preds, 1), entries are unique and in {0, 1,..., num_anchors})
        keep_inds = keep_inds.squeeze() # (shape (num_foreground_preds, ), entries are unique and in {0, 1,..., num_anchors})

        # get all pred non-background bboxes:
        outputs_regr = outputs_regr[keep_inds] # (shape (num_foreground_preds, 4), (x, y, w, h))
        anchor_bboxes = self.anchor_bboxes[keep_inds] # (shape (num_foreground_preds, 4), (x, y, w, h))
        pred_max_scores = pred_max_scores[keep_inds] # (shape (num_foreground_preds, ))
        pred_class_labels = pred_class_labels[keep_inds] # (shape (num_foreground_preds, ))

        # print ("Number of predicted bboxes before thresholding:")
        # print (outputs_regr.size())

        if outputs_regr.size() == torch.Size([4]):
            outputs_regr = outputs_regr.unsqueeze(0)
            anchor_bboxes = anchor_bboxes.unsqueeze(0)
            pred_max_scores = torch.from_numpy(np.array([pred_max_scores.data]))
            pred_class_labels = torch.from_numpy(np.array([pred_class_labels.data]))

        if outputs_regr.size(0) > 0:
            # get the indices for all pred bboxes with a large enough pred class score:
            keep_inds = pred_max_scores > self.conf_thresh # (shape (num_foreground_preds, ), entries in {0, 1})
            keep_inds = keep_inds.nonzero() # (shape (num_preds_before_nms, 1), entries are unique and in {0, 1,..., num_foreground_preds})
            keep_inds = keep_inds.squeeze() # (shape (num_preds_before_nms, ), entries are unique and in {0, 1,..., num_foreground_preds})

            # get all pred bboxes with a large enough pred class score:
            outputs_regr = outputs_regr[keep_inds] # (shape (num_preds_before_nms, 4), (x, y, w, h))
            anchor_bboxes = anchor_bboxes[keep_inds] # (shape (num_preds_before_nms, 4), (x, y, w, h))
            pred_max_scores = pred_max_scores[keep_inds] # (shape (num_preds_before_nms, ))
            pred_class_labels = pred_class_labels[keep_inds] # (shape (num_preds_before_nms, ))

            # print ("Number of predicted bboxes before NMS:")
            # print (outputs_regr.size())

            if outputs_regr.size() == torch.Size([4]):
                outputs_regr = outputs_regr.unsqueeze(0)
                anchor_bboxes = anchor_bboxes.unsqueeze(0)
                pred_max_scores = torch.from_numpy(np.array([pred_max_scores.data]))
                pred_class_labels = torch.from_numpy(np.array([pred_class_labels.data]))

            if outputs_regr.size(0) > 0:
                # pred_x = anchor_w*output_x + anchor_x:
                pred_x = anchor_bboxes[:, 2]*outputs_regr[:, 0] + anchor_bboxes[:, 0] # (shape (num_anchors, ))
                pred_x = pred_x.view(-1, 1) # (shape (num_anchors, 1))
                # pred_y = anchor_h*output_y + anchor_y:
                pred_y = anchor_bboxes[:, 3]*outputs_regr[:, 1] + anchor_bboxes[:, 1]
                pred_y = pred_y.view(-1, 1)
                # pred_w = exp(output_w)*anchor_w:
                pred_w = torch.exp(outputs_regr[:, 2])*anchor_bboxes[:, 2]
                pred_w = pred_w.view(-1, 1)
                # pred_h = exp(output_h)*anchor_h:
                pred_h = torch.exp(outputs_regr[:, 3])*anchor_bboxes[:, 3]
                pred_h = pred_h.view(-1, 1)

                pred_bboxes = torch.cat([pred_x, pred_y, pred_w, pred_h], 1) # (shape (num_preds_before_nms, 4), (x, y, w, h))

                # filter bboxes by performing nms:
                keep_inds = self._bbox_nms(pred_bboxes, pred_max_scores) # (shape: (num_preds_after_nms, ))
                pred_bboxes = pred_bboxes[keep_inds]
                pred_max_scores = pred_max_scores[keep_inds]
                pred_class_labels = pred_class_labels[keep_inds]

                # (pred_bboxes has shape (num_preds_after_nms, 4), (x, y, w, h))
                # (pred_max_scores has shape (num_preds_after_nms, ))
                # (pred_class_labels has shape (num_preds_after_nms, ))
                return (pred_bboxes, pred_max_scores, pred_class_labels)
            else:
                #print ("None!")
                return (None, None, None)
        else:
            return (None, None, None)

    def decode_gt_single(self, labels_regr):
        # (labels_regr has shape (num_anchors, 4), (x, y, w, h))
        # (self.anchor_bboxes has shape (num_anchors, 4), (x, y, w, h))

        # gt_x = anchor_w*label_x + anchor_x:
        gt_x = self.anchor_bboxes[:, 2]*labels_regr[:, 0] + self.anchor_bboxes[:, 0] # (shape (num_anchors, ))
        gt_x = gt_x.view(-1, 1) # (shape (num_anchors, 1))
        # gt_y = anchor_h*label_y + anchor_y:
        gt_y = self.anchor_bboxes[:, 3]*labels_regr[:, 1] + self.anchor_bboxes[:, 1]
        gt_y = gt_y.view(-1, 1) # (shape (num_anchors, 1))
        # gt_w = exp(label_w)*anchor_w:
        gt_w = torch.exp(labels_regr[:, 2])*self.anchor_bboxes[:, 2]
        gt_w = gt_w.view(-1, 1) # (shape (num_anchors, 1))
        # gt_h = exp(label_h)*anchor_h:
        gt_h = torch.exp(labels_regr[:, 3])*self.anchor_bboxes[:, 3]
        gt_h = gt_h.view(-1, 1) # (shape (num_anchors, 1))

        gt_bboxes = torch.cat([gt_x, gt_y, gt_w, gt_h], 1) # (shape (num_anchors, 4), (x, y, w, h))

        return gt_bboxes

# bbox_encoder = BboxEncoder()
# bbox_encoder.encode(torch.Tensor([[600, 800, 300, 400], [640, 810, 200, 300]]), torch.Tensor([1, 2]))
# bbox_encoder.decode(torch.ones((bbox_encoder.num_anchors, 4)), torch.ones((bbbox_encoder.num_anchors, 4)))

class DatasetAugmentation(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        self.img_height = 375
        self.img_width = 1242

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (car, pedestrian, cyclist, background)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            labels = LabelLoader2D3D(img_id, self.label_dir, ".txt", self.calib_dir, ".txt")

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

        gt_bboxes = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))

        ########################################################################
        # flip the img and the labels with 0.5 probability:
        ########################################################################
        flip = np.random.randint(low=0, high=2)
        if flip == 1:
            img = cv2.flip(img, 1)

            img_w = self.img_width
            gt_bboxes[:, 0:2] = img_w - gt_bboxes[:, 0:2]
            temp = np.copy(gt_bboxes[:, 0])
            gt_bboxes[:, 0] = gt_bboxes[:, 1]
            gt_bboxes[:, 1] = temp

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes.shape[0]):
        #     bbox = gt_bboxes[i]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes = torch.from_numpy(gt_bboxes) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(example["class_labels"]) # (shape (num_gt_objects, ))
        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class)

    def __len__(self):
        return self.num_examples

# test = DatasetAugmentation("/home/fregu856/exjobb/data/kitti", "/home/fregu856/exjobb/data/kitti/meta", type="train")
# for i in range(10):
#     _ = test.__getitem__(i)

class DatasetMoreAugmentation2(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        self.img_height = 375
        self.img_width = 1242

        self.random_horizontal_flip = RandomHorizontalFlip(p=0.5)
        self.random_hsv = RandomHSV(hue=10, saturation=20, brightness=20)
        self.random_scale = RandomScale(scale=0.3)
        self.random_translate = RandomTranslate(translate=0.2)

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (car, pedestrian, cyclist, background)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            labels = LabelLoader2D3D(img_id, self.label_dir, ".txt", self.calib_dir, ".txt")

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

        gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

        gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
        gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
        gt_bboxes_xxyyc[:, 4] = gt_classes

        ########################################################################
        # data augmentation START:
        ########################################################################
        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyy.shape[0]):
        #     bbox = gt_bboxes_xxyy[i]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # flip the img and the labels with 0.5 probability:
        img, gt_bboxes_xyxyc = self.random_horizontal_flip(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly modify the hue, saturation and brightness of the image:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv, gt_bboxes_xyxyc = self.random_hsv(img_hsv, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # scale the image and the labels with a factor drawn from Uniform[1-scale, 1+scale]:
        img, gt_bboxes_xyxyc = self.random_scale(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly translate the image and the labels:
        img, gt_bboxes_xyxyc = self.random_translate(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #
        ########################################################################
        # data augmentation END:
        ########################################################################

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes_xxyy = gt_bboxes_xxyyc[:, 0:4] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = gt_bboxes_xxyyc[:, 4] # (shape (num_gt_objects, ))
        gt_bboxes_xxyy = torch.from_numpy(gt_bboxes_xxyy) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(gt_classes) # (shape (num_gt_objects, ))

        if gt_bboxes_xxyy.size(0) == 0: # (if 0 gt objects)
            return self.__getitem__(index+1)

        if gt_bboxes_xxyy.size() == torch.Size([4]): # (if 1 gt object)
            gt_bboxes_xxyy = gt_bboxes_xxyy.unsqueeze(0)
            gt_classes = torch.from_numpy(np.array([gt_classes.data]))

        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes_xxyy, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class)

    def __len__(self):
        return self.num_examples

class DatasetEval(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, type):
        self.img_dir = kitti_data_path + "/object/training/image_2/"
        self.label_dir = kitti_data_path + "/object/training/label_2/"
        self.calib_dir = kitti_data_path + "/object/training/calib/"

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        self.img_height = 375
        self.img_width = 1242

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (car, pedestrian, cyclist, background)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            labels = LabelLoader2D3D(img_id, self.label_dir, ".txt", self.calib_dir, ".txt")

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

        gt_bboxes = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes.shape[0]):
        #     bbox = gt_bboxes[i]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes = torch.from_numpy(gt_bboxes) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(example["class_labels"]) # (shape (num_gt_objects, ))
        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class, img_id)

    def __len__(self):
        return self.num_examples

class DatasetEvalSeq(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, sequence):
        self.img_dir = kitti_data_path + "/tracking/training/image_02/" + sequence + "/"
        self.label_path = kitti_data_path + "/tracking/training/label_02/" + sequence + ".txt"
        self.calib_path = kitti_meta_path + "/tracking/training/calib/" + sequence + ".txt" # NOTE! NOTE! the data format for the calib files was sliightly different for tracking, so I manually modifed the 20 files and saved them in the kitti_meta folder

        self.img_height = 375
        self.img_width = 1242

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (car, pedestrian, cyclist, background)

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            if img_id.lstrip('0') == '':
                img_id_float = 0.0
            else:
                img_id_float = float(img_id.lstrip('0'))

            labels = LabelLoader2D3D_sequence(img_id, img_id_float, self.label_path, self.calib_path)

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

        gt_bboxes = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))

        if gt_bboxes.shape[0] == 0:
            return self.__getitem__(index-1)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes.shape[0]):
        #     bbox = gt_bboxes[i]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes = torch.from_numpy(gt_bboxes) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(example["class_labels"]) # (shape (num_gt_objects, ))
        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class, img_id)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path):
        self.img_dir = kitti_data_path + "/object/testing/image_2/"

        self.img_height = 375
        self.img_width = 1242

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = img_ids

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        img_id = self.examples[index]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

        # # # # # # debug visualization:
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # # # # # #

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        return (img, img_id)

    def __len__(self):
        return self.num_examples

class DatasetTestSeq(torch.utils.data.Dataset):
    def __init__(self, kitti_data_path, kitti_meta_path, sequence):
        self.img_dir = kitti_data_path + "/tracking/testing/image_02/" + sequence + "/"

        self.img_height = 375
        self.img_width = 1242

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = img_ids

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        img_id = self.examples[index]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

        # # # # # # debug visualization:
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        # # # # # #

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        return (img, img_id)

    def __len__(self):
        return self.num_examples

class DatasetThnSeq(torch.utils.data.Dataset):
    def __init__(self, thn_data_path):
        self.img_dir = thn_data_path + "/"

        self.orig_img_height = 512
        self.orig_img_width = 1024

        self.img_height = 375
        self.img_width = 1242

        self.examples = []

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = img_ids

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        img_id = self.examples[index]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, int((self.img_width/self.orig_img_width)*self.img_height)))
        img = img[(int((self.img_width/self.orig_img_width)*self.img_height) - self.img_height):int((self.img_width/self.orig_img_width)*self.img_height)]

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        return (img, img_id)

    def __len__(self):
        return self.num_examples

class DatasetThnSeqSynscapes(torch.utils.data.Dataset):
    def __init__(self, thn_data_path):
        self.img_dir = thn_data_path + "/"

        self.orig_img_height = 512
        self.orig_img_width = 1024

        self.img_height = 720
        self.img_width = 1440

        self.examples = []

        img_ids = []
        img_names = os.listdir(self.img_dir)
        for img_name in img_names:
            img_id = img_name.split(".png")[0]
            img_ids.append(img_id)

        self.examples = img_ids

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        img_id = self.examples[index]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1)
        img = cv2.resize(img, (self.img_width, self.img_height))

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        return (img, img_id)

    def __len__(self):
        return self.num_examples

from synscapesloader import LabelLoader2D3D_Synscapes

class_string_to_label_synscapes = {"car": 1,
                                   "person": 2,
                                   "bicyclist": 3} # (background: 0)

class DatasetSynscapesAugmentation(torch.utils.data.Dataset):
    def __init__(self, synscapes_path, synscapes_meta_path, type):
        self.img_dir = synscapes_path + "/img/rgb/"
        self.meta_dir = synscapes_path + "/meta/"

        with open(synscapes_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        self.orig_img_height = 720
        self.orig_img_width = 1440

        self.img_height = 375
        self.img_width = 1242

        self.random_horizontal_flip = RandomHorizontalFlip(p=0.5)
        self.random_hsv = RandomHSV(hue=10, saturation=20, brightness=20)
        self.random_scale = RandomScale(scale=0.3)
        self.random_translate = RandomTranslate(translate=0.2)

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (background, car, pedestrian, cyclist)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            labels = LabelLoader2D3D_Synscapes(meta_dir=self.meta_dir, file_id=img_id)

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["car", "person", "bicyclist"] and label_2d["occluded"] < 0.7 and label_2d["truncated"] < 0.7:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label_synscapes[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1) # (shape: (orig_img_height, orig_img_width, 3))

        gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

        gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
        gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
        gt_bboxes_xxyyc[:, 4] = gt_classes

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        scale = float(float(self.img_width)/float(self.orig_img_width))
        img = cv2.resize(img, (self.img_width, int(scale*self.orig_img_height))) # (shape: (621, img_width, 3))
        gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyyc[:, 0:4]*scale

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        start = int(np.random.uniform(low=0, high=(img.shape[0] - self.img_height)))
        img = img[start:(start + self.img_height)] # (shape: (img_height, img_width, 3))
        gt_bboxes_xxyyc[:, 2:4] = gt_bboxes_xxyyc[:, 2:4] - start

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        ########################################################################
        # data augmentation START:
        ########################################################################
        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # flip the img and the labels with 0.5 probability:
        img, gt_bboxes_xyxyc = self.random_horizontal_flip(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly modify the hue, saturation and brightness of the image:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv, gt_bboxes_xyxyc = self.random_hsv(img_hsv, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # scale the image and the labels with a factor drawn from Uniform[1-scale, 1+scale]:
        img, gt_bboxes_xyxyc = self.random_scale(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly translate the image and the labels:
        img, gt_bboxes_xyxyc = self.random_translate(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #
        ########################################################################
        # data augmentation END:
        ########################################################################

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes_xxyy = gt_bboxes_xxyyc[:, 0:4] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = gt_bboxes_xxyyc[:, 4] # (shape (num_gt_objects, ))
        gt_bboxes_xxyy = torch.from_numpy(gt_bboxes_xxyy) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(gt_classes) # (shape (num_gt_objects, ))

        if gt_bboxes_xxyy.size(0) == 0: # (if 0 gt objects)
            return self.__getitem__(index+1)

        if gt_bboxes_xxyy.size() == torch.Size([4]): # (if 1 gt object)
            gt_bboxes_xxyy = gt_bboxes_xxyy.unsqueeze(0)
            gt_classes = torch.from_numpy(np.array([gt_classes.data]))

        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes_xxyy, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class)

    def __len__(self):
        return self.num_examples

# test = DatasetSynscapesAugmentation("/home/fregu856/data/synscapes", "", "")
# for i in range(10):
#     _ = test.__getitem__(i)

class DatasetKITTISynscapesAugmentation(torch.utils.data.Dataset):
    def __init__(self, synscapes_path, synscapes_meta_path, kitti_data_path, kitti_meta_path, type):
        self.synscapes_img_dir = synscapes_path + "/img/rgb/"
        self.synscapes_meta_dir = synscapes_path + "/meta/"

        with open(synscapes_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            synscapes_img_ids = pickle.load(file)

        self.kitti_img_dir = kitti_data_path + "/object/training/image_2/"
        self.kitti_label_dir = kitti_data_path + "/object/training/label_2/"
        self.kitti_calib_dir = kitti_data_path + "/object/training/calib/"
        self.kitti_lidar_dir = kitti_data_path + "/object/training/velodyne/"

        with open(kitti_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            kitti_img_ids = pickle.load(file)

        num_kitti_imgs = len(kitti_img_ids)
        synscapes_img_ids = synscapes_img_ids[0:num_kitti_imgs]

        self.synscapes_img_height = 720
        self.synscapes_img_width = 1440

        self.img_height = 375
        self.img_width = 1242

        self.random_horizontal_flip = RandomHorizontalFlip(p=0.5)
        self.random_hsv = RandomHSV(hue=10, saturation=20, brightness=20)
        self.random_scale = RandomScale(scale=0.3)
        self.random_translate = RandomTranslate(translate=0.2)

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (background, car, pedestrian, cyclist)

        self.examples = []

        for img_id in synscapes_img_ids:
            example = {}
            example["dataset"] = "synscapes"
            example["img_id"] = img_id

            labels = LabelLoader2D3D_Synscapes(meta_dir=self.synscapes_meta_dir, file_id=img_id)

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["car", "person", "bicyclist"] and label_2d["occluded"] < 0.7 and label_2d["truncated"] < 0.7:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label_synscapes[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        for img_id in kitti_img_ids:
            example = {}
            example["dataset"] = "kitti"
            example["img_id"] = img_id

            labels = LabelLoader2D3D(img_id, self.kitti_label_dir, ".txt", self.kitti_calib_dir, ".txt")

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        if example["dataset"] == "synscapes":
            img_id = example["img_id"]

            img_path = self.synscapes_img_dir + img_id + ".png"
            img = cv2.imread(img_path, -1) # (shape: (orig_img_height, orig_img_width, 3))

            gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
            gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

            if gt_classes.shape[0] == 0:
                return self.__getitem__(0)

            gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
            gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
            gt_bboxes_xxyyc[:, 4] = gt_classes

            scale = float(float(self.img_width)/float(self.synscapes_img_width))
            img = cv2.resize(img, (self.img_width, int(scale*self.synscapes_img_height))) # (shape: (621, img_width, 3))
            gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyyc[:, 0:4]*scale

            start = int(np.random.uniform(low=0, high=(img.shape[0] - self.img_height)))
            img = img[start:(start + self.img_height)] # (shape: (img_height, img_width, 3))
            gt_bboxes_xxyyc[:, 2:4] = gt_bboxes_xxyyc[:, 2:4] - start

            # # # # # # debug visualization:
            # bbox_polys = []
            # for i in range(gt_bboxes_xxyyc.shape[0]):
            #     bbox = gt_bboxes_xxyyc[i, 0:4]
            #     bbox_poly = create2Dbbox_poly(bbox)
            #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
            #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
            #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
            #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
            #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
            #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
            #     bbox_polys.append(bbox_poly)
            # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
            # cv2.imshow("test", img_with_gt_bboxes)
            # cv2.waitKey(0)
            # # # # # #

        elif example["dataset"] == "kitti":
            img_id = example["img_id"]

            img_path = self.kitti_img_dir + img_id + ".png"
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

            gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
            gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

            if gt_classes.shape[0] == 0:
                return self.__getitem__(0)

            gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
            gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
            gt_bboxes_xxyyc[:, 4] = gt_classes

        ########################################################################
        # data augmentation START:
        ########################################################################
        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # flip the img and the labels with 0.5 probability:
        img, gt_bboxes_xyxyc = self.random_horizontal_flip(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly modify the hue, saturation and brightness of the image:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv, gt_bboxes_xyxyc = self.random_hsv(img_hsv, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # scale the image and the labels with a factor drawn from Uniform[1-scale, 1+scale]:
        img, gt_bboxes_xyxyc = self.random_scale(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly translate the image and the labels:
        img, gt_bboxes_xyxyc = self.random_translate(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     if gt_bboxes_xxyyc[i, 4] == 1: # (Car)
        #         bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 2: # (Pedestrian)
        #         bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
        #     elif gt_bboxes_xxyyc[i, 4] == 3: # (Cyclist)
        #         bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #
        ########################################################################
        # data augmentation END:
        ########################################################################

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes_xxyy = gt_bboxes_xxyyc[:, 0:4] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = gt_bboxes_xxyyc[:, 4] # (shape (num_gt_objects, ))
        gt_bboxes_xxyy = torch.from_numpy(gt_bboxes_xxyy) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(gt_classes) # (shape (num_gt_objects, ))

        if gt_bboxes_xxyy.size(0) == 0: # (if 0 gt objects)
            return self.__getitem__(index+1)

        if gt_bboxes_xxyy.size() == torch.Size([4]): # (if 1 gt object)
            gt_bboxes_xxyy = gt_bboxes_xxyy.unsqueeze(0)
            gt_classes = torch.from_numpy(np.array([gt_classes.data]))

        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes_xxyy, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class)

    def __len__(self):
        return self.num_examples

class DatasetSynscapesEval(torch.utils.data.Dataset):
    def __init__(self, synscapes_path, synscapes_meta_path, type):
        self.img_dir = synscapes_path + "/img/rgb/"
        self.meta_dir = synscapes_path + "/meta/"

        with open(synscapes_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        self.orig_img_height = 720
        self.orig_img_width = 1440

        self.img_height = 375
        self.img_width = 1242

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (background, car, pedestrian, cyclist)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            labels = LabelLoader2D3D_Synscapes(meta_dir=self.meta_dir, file_id=img_id)

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["car", "person", "bicyclist"] and label_2d["occluded"] < 0.7 and label_2d["truncated"] < 0.7:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label_synscapes[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1) # (shape: (orig_img_height, orig_img_width, 3))

        gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

        gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
        gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
        gt_bboxes_xxyyc[:, 4] = gt_classes

        scale = float(float(self.img_width)/float(self.orig_img_width))
        img = cv2.resize(img, (self.img_width, int(scale*self.orig_img_height))) # (shape: (621, img_width, 3))
        gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyyc[:, 0:4]*scale

        start = img.shape[0] - self.img_height
        img = img[start:(start + self.img_height)] # (shape: (img_height, img_width, 3))
        gt_bboxes_xxyyc[:, 2:4] = gt_bboxes_xxyyc[:, 2:4] - start

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes_xxyy = gt_bboxes_xxyyc[:, 0:4] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = gt_bboxes_xxyyc[:, 4] # (shape (num_gt_objects, ))
        gt_bboxes_xxyy = torch.from_numpy(gt_bboxes_xxyy) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(gt_classes) # (shape (num_gt_objects, ))

        if gt_bboxes_xxyy.size(0) == 0: # (if 0 gt objects)
            return self.__getitem__(index+1)

        if gt_bboxes_xxyy.size() == torch.Size([4]): # (if 1 gt object)
            gt_bboxes_xxyy = gt_bboxes_xxyy.unsqueeze(0)
            gt_classes = torch.from_numpy(np.array([gt_classes.data]))

        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes_xxyy, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class, img_id)

    def __len__(self):
        return self.num_examples

class DatasetSynscapesEvalFullSize(torch.utils.data.Dataset):
    def __init__(self, synscapes_path, synscapes_meta_path, type):
        self.img_dir = synscapes_path + "/img/rgb/"
        self.meta_dir = synscapes_path + "/meta/"

        with open(synscapes_meta_path + "/%s_img_ids.pkl" % type, "rb") as file: # (needed for python3)
            img_ids = pickle.load(file)

        self.img_height = 720
        self.img_width = 1440

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (background, car, pedestrian, cyclist)

        self.examples = []
        for img_id in img_ids:
            example = {}
            example["img_id"] = img_id

            labels = LabelLoader2D3D_Synscapes(meta_dir=self.meta_dir, file_id=img_id)

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["car", "person", "bicyclist"] and label_2d["occluded"] < 0.7 and label_2d["truncated"] < 0.7:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label_synscapes[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_id = example["img_id"]

        img_path = self.img_dir + img_id + ".png"
        img = cv2.imread(img_path, -1) # (shape: (orig_img_height, orig_img_width, 3))

        gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

        gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
        gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
        gt_bboxes_xxyyc[:, 4] = gt_classes

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes_xxyy = gt_bboxes_xxyyc[:, 0:4] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = gt_bboxes_xxyyc[:, 4] # (shape (num_gt_objects, ))
        gt_bboxes_xxyy = torch.from_numpy(gt_bboxes_xxyy) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(gt_classes) # (shape (num_gt_objects, ))

        if gt_bboxes_xxyy.size(0) == 0: # (if 0 gt objects)
            return self.__getitem__(index+1)

        if gt_bboxes_xxyy.size() == torch.Size([4]): # (if 1 gt object)
            gt_bboxes_xxyy = gt_bboxes_xxyy.unsqueeze(0)
            gt_classes = torch.from_numpy(np.array([gt_classes.data]))

        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes_xxyy, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class, img_id)

    def __len__(self):
        return self.num_examples











def ProjectTo2Dbbox(bbox_3D, cam_intrinsic, R):
    # (bbox_3D is an array of shape: (6, ), (x, y, z, h, w, l) in cam coordinates)
    # (cam_intrinsic is an array of shape: (3, 3))

    x = bbox_3D[0]
    y = bbox_3D[1]
    z = bbox_3D[2]
    h = bbox_3D[3]
    w = bbox_3D[4]
    l = bbox_3D[5]

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = (l/2.0)*np.array([1,  1,  1,  1, -1, -1, -1, -1])
    y_corners = (w/2.0)*np.array([1, -1, -1,  1,  1, -1, -1,  1])
    z_corners = (h/2.0)*np.array([1,  1, -1, -1,  1,  1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(R, corners)

    # Translate
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic

    points = corners
    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    points = points/points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    points = points.T

    u_min = np.min(points[:, 0])
    v_min = np.min(points[:, 1])
    u_max = np.max(points[:, 0])
    v_max = np.max(points[:, 1])

    left = int(u_min)
    top = int(v_min)
    right = int(u_max)
    bottom = int(v_max)

    projected_2Dbbox = [left, top, right, bottom]

    return projected_2Dbbox

def wrapToPi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

sys.path.append("/root/project1/nuscenes-devkit/python-sdk")
sys.path.append("/home/fregu856/project1/nuscenes-devkit/python-sdk")
from nuscenes_utils.nuscenes import NuScenes
from nuscenes_utils.data_classes import PointCloud as NuScenesPointCloud

from pyquaternion import Quaternion

class_string_to_label_nuscenes = {"vehicle.car": 1,
                                  "human.pedestrian.adult": 2,
                                  "human.pedestrian.child": 2,
                                  "human.pedestrian.police_officer": 2,
                                  "human.pedestrian.construction_worker": 2,
                                  "vehicle.bicycle": 3} # (background: 0)

class DatasetKITTINuscenesAugmentation(torch.utils.data.Dataset):
    def __init__(self, nuscenes_data_path, kitti_data_path, kitti_meta_path, kitti_type):
        self.nuscenes_data_path = nuscenes_data_path
        self.nusc = NuScenes(version="v0.1", dataroot=self.nuscenes_data_path, verbose=False)

        self.kitti_img_dir = kitti_data_path + "/object/training/image_2/"
        self.kitti_label_dir = kitti_data_path + "/object/training/label_2/"
        self.kitti_calib_dir = kitti_data_path + "/object/training/calib/"
        self.kitti_lidar_dir = kitti_data_path + "/object/training/velodyne/"

        with open(kitti_meta_path + "/%s_img_ids.pkl" % kitti_type, "rb") as file: # (needed for python3)
            kitti_img_ids = pickle.load(file)

        self.img_height = 375
        self.img_width = 1242

        self.nuscenes_img_height = 900
        self.nuscenes_img_width = 1600

        self.random_horizontal_flip = RandomHorizontalFlip(p=0.5)
        self.random_hsv = RandomHSV(hue=10, saturation=20, brightness=20)
        self.random_scale = RandomScale(scale=0.3)
        self.random_translate = RandomTranslate(translate=0.2)

        self.bbox_encoder = BboxEncoder(img_h=self.img_height, img_w=self.img_width)

        self.num_classes = 4 # (car, pedestrian, cyclist, background)

        self.examples = []

        samples = self.nusc.sample
        for sample in samples:
            example = {}
            example["dataset"] = "nuscenes"
            example["sample_token"] = sample["token"]
            bboxes = []
            class_labels = []

            cam_front_token = sample["data"]["CAM_FRONT"]
            cam_front = self.nusc.get("sample_data", cam_front_token)

            sd_record = cam_front
            pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

            cs_record = self.nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
            cam_front_intrinsic = np.array(cs_record['camera_intrinsic']) # (shape: (3, 3))

            annotation_tokens = sample["anns"]
            for annotation_token in annotation_tokens:
                annotation = self.nusc.get("sample_annotation", annotation_token)
                if annotation["category_name"] in ["vehicle.car", "human.pedestrian.adult", "human.pedestrian.police_officer", "human.pedestrian.child", "human.pedestrian.construction_worker", "vehicle.bicycle"]:
                    if int(annotation["visibility_token"]) > 1:
                        translation = annotation["translation"] # (X, Y, Z) (global frame)

                        r_y_quat = Quaternion(annotation["rotation"])

                        # transform from global frame into the ego vehicle frame for the timestamp of the front-camera image:
                        translation = translation - np.array(pose_record['translation'])
                        translation = np.dot(Quaternion(pose_record['rotation']).inverse.rotation_matrix, translation)
                        r_y_quat = Quaternion(pose_record['rotation']).inverse*r_y_quat

                        # transform into the front-camera frame (same as the point cloud is transformed into):
                        translation = translation - np.array(cs_record['translation'])
                        translation = np.dot(Quaternion(cs_record['rotation']).inverse.rotation_matrix, translation)
                        r_y_quat = Quaternion(cs_record['rotation']).inverse*r_y_quat

                        size = annotation["size"] # (w, l, h)

                        bbox_3D = np.array([translation[0], translation[1], translation[2], size[2], size[0], size[1]]) # (x, y, z, h, w, l)

                        R = r_y_quat.rotation_matrix

                        if bbox_3D[2] > 2.0: # (remove all bboxes which are located behind the camera)
                            bbox_xyxy = ProjectTo2Dbbox(bbox_3D, cam_front_intrinsic, R) # (x_min, y_min, x_max, y_max)
                            bbox = [bbox_xyxy[0], bbox_xyxy[2], bbox_xyxy[1], bbox_xyxy[3]] # (x_min, x_max, y_min, y_max)

                            bboxes.append(bbox)
                            class_labels.append(class_string_to_label_nuscenes[annotation["category_name"]])

            example["bboxes"] = np.array(bboxes, dtype=np.float32)
            example["class_labels"] = np.array(class_labels, dtype=np.float32)
            self.examples.append(example)

        for img_id in kitti_img_ids:
            example = {}
            example["dataset"] = "kitti"
            example["img_id"] = img_id

            labels = LabelLoader2D3D(img_id, self.kitti_label_dir, ".txt", self.kitti_calib_dir, ".txt")

            bboxes = np.zeros((len(labels), 4), dtype=np.float32)
            class_labels = np.zeros((len(labels), ), dtype=np.float32)
            counter = 0
            for label in labels:
                label_2d = label["label_2D"]
                if label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
                    bbox = label_2d["poly"]
                    u_min = bbox[0, 0] # (left)
                    u_max = bbox[1, 0] # (rigth)
                    v_min = bbox[0, 1] # (top)
                    v_max = bbox[2, 1] # (bottom)
                    bboxes[counter] = np.array([u_min, u_max, v_min, v_max])

                    class_labels[counter] = class_string_to_label[label_2d["class"]]

                    counter += 1

            bboxes = bboxes[0:counter]
            class_labels = class_labels[0:counter]

            example["bboxes"] = bboxes
            example["class_labels"] = class_labels
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        if example["dataset"] == "kitti":
            img_id = example["img_id"]

            img_path = self.kitti_img_dir + img_id + ".png"
            img = cv2.imread(img_path, -1)
            img = cv2.resize(img, (self.img_width, self.img_height)) # (shape: (img_height, img_width, 3))

            gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
            gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

            if gt_classes.shape[0] == 0:
                return self.__getitem__(0)

            gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
            gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
            gt_bboxes_xxyyc[:, 4] = gt_classes

        elif example["dataset"] == "nuscenes":
            sample_token = example["sample_token"]

            sample = self.nusc.get("sample", sample_token)

            cam_front_token = sample["data"]["CAM_FRONT"]
            cam_front_sample_data = self.nusc.get("sample_data", cam_front_token)
            cam_front_filename = cam_front_sample_data["filename"]
            img_path = self.nuscenes_data_path + "/" + cam_front_filename
            img = cv2.imread(img_path, -1)

            gt_bboxes_xxyy = example["bboxes"] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
            gt_classes = example["class_labels"] # (shape: (num_gt_objects, ))

            if gt_classes.shape[0] == 0:
                return self.__getitem__(0)

            gt_bboxes_xxyyc = np.zeros((gt_bboxes_xxyy.shape[0], 5), dtype=gt_bboxes_xxyy.dtype) # (shape: (num_gt_objects, 5), (x_min, x_max, y_min, y_max, class_label))
            gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyy
            gt_bboxes_xxyyc[:, 4] = gt_classes

            scale = float(float(self.img_width)/float(self.nuscenes_img_width))
            img = cv2.resize(img, (self.img_width, int(scale*self.nuscenes_img_height))) # (shape: (621, img_width, 3))
            gt_bboxes_xxyyc[:, 0:4] = gt_bboxes_xxyyc[:, 0:4]*scale

            start = int(np.random.uniform(low=0, high=(img.shape[0] - self.img_height)))
            img = img[start:(start + self.img_height)] # (shape: (img_height, img_width, 3))
            gt_bboxes_xxyyc[:, 2:4] = gt_bboxes_xxyyc[:, 2:4] - start

        ########################################################################
        # data augmentation START:
        ########################################################################
        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # flip the img and the labels with 0.5 probability:
        img, gt_bboxes_xyxyc = self.random_horizontal_flip(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly modify the hue, saturation and brightness of the image:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv, gt_bboxes_xyxyc = self.random_hsv(img_hsv, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # scale the image and the labels with a factor drawn from Uniform[1-scale, 1+scale]:
        img, gt_bboxes_xyxyc = self.random_scale(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #

        # randomly translate the image and the labels:
        img, gt_bboxes_xyxyc = self.random_translate(img, bboxes_xxyyc_2_xyxyc(gt_bboxes_xxyyc))
        gt_bboxes_xxyyc = bboxes_xyxyc_2_xxyyc(gt_bboxes_xyxyc)

        # # # # # # debug visualization:
        # bbox_polys = []
        # for i in range(gt_bboxes_xxyyc.shape[0]):
        #     bbox = gt_bboxes_xxyyc[i, 0:4]
        #     bbox_poly = create2Dbbox_poly(bbox)
        #     bbox_polys.append(bbox_poly)
        # img_with_gt_bboxes = draw_2d_polys_no_text(img, bbox_polys)
        # cv2.imshow("test", img_with_gt_bboxes)
        # cv2.waitKey(0)
        # # # # # #
        ########################################################################
        # data augmentation END:
        ########################################################################

        ########################################################################
        # normalize the img:
        ########################################################################
        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225]) # (shape: (img_height, img_width, 3))
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, img_height, img_width))
        img = img.astype(np.float32)

        ########################################################################
        # get ground truth:
        ########################################################################
        gt_bboxes_xxyy = gt_bboxes_xxyyc[:, 0:4] # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = gt_bboxes_xxyyc[:, 4] # (shape (num_gt_objects, ))
        gt_bboxes_xxyy = torch.from_numpy(gt_bboxes_xxyy) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
        gt_classes = torch.from_numpy(gt_classes) # (shape (num_gt_objects, ))

        if gt_bboxes_xxyy.size(0) == 0: # (if 0 gt objects)
            return self.__getitem__(index+1)

        if gt_bboxes_xxyy.size() == torch.Size([4]): # (if 1 gt object)
            gt_bboxes_xxyy = gt_bboxes_xxyy.unsqueeze(0)
            gt_classes = torch.from_numpy(np.array([gt_classes.data]))

        label_regr, label_class = self.bbox_encoder.encode(gt_bboxes_xxyy, gt_classes)
        # (label_regr is a Tensor of shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class is a Tensor of shape: (num_anchors, ))

        ########################################################################
        # convert numpy -> torch:
        ########################################################################
        img = torch.from_numpy(img) # (shape: (3, img_height, img_width))

        # (img has shape: (3, img_height, img_width))
        # (label_regr has shape: (num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        # (label_class has shape: (num_anchors, ))
        return (img, label_regr, label_class)

    def __len__(self):
        return self.num_examples

# test = DatasetKITTINuscenesAugmentation("/home/fregu856/project1/data/nuscenes", "/home/fregu856/exjobb/data/kitti", "/home/fregu856/exjobb/data/kitti/meta", "train")
# for i in range(200):
#     _ = test.__getitem__(i)
# # _ = test.__getitem__(0)
