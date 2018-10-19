# from kittiloader import LabelLoader2D3D # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
#
# import torch
# import torch.utils.data
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# import pickle
# import numpy as np
# import cv2
# import math
#
# from datasets import BboxEncoder, class_string_to_label
#
# ################################################################################
# # compute the class weigths:
# ################################################################################
# # (0: background, 1: car, 2: pedestrian, 3: cyclist)
#
# kitti_data_path = "/root/3DOD_thesis/data/kitti"
# kitti_meta_path = "/root/retinanet/data/kitti/meta"
#
# img_dir = kitti_data_path + "/object/training/image_2/"
# label_dir = kitti_data_path + "/object/training/label_2/"
# calib_dir = kitti_data_path + "/object/training/calib/"
#
# with open(kitti_meta_path + "/train_img_ids.pkl", "rb") as file: # (needed for python3)
#     train_img_ids = pickle.load(file)
#
# img_height = 375
# img_width = 1242
#
# bbox_encoder = BboxEncoder(img_h=img_height, img_w=img_width)
#
# num_classes = 4 # (car, pedestrian, cyclist, background)
#
# examples = []
# for img_id in train_img_ids:
#     example = {}
#     example["img_id"] = img_id
#
#     labels = LabelLoader2D3D(img_id, label_dir, ".txt", calib_dir, ".txt")
#
#     bboxes = np.zeros((len(labels), 4), dtype=np.float32)
#     class_labels = np.zeros((len(labels), ), dtype=np.float32)
#     counter = 0
#     for label in labels:
#         label_2d = label["label_2D"]
#         if label_2d["truncated"] <= 0.50 and label_2d["class"] in ["Car", "Pedestrian", "Cyclist"]:
#             bbox = label_2d["poly"]
#             u_min = bbox[0, 0] # (left)
#             u_max = bbox[1, 0] # (rigth)
#             v_min = bbox[0, 1] # (top)
#             v_max = bbox[2, 1] # (bottom)
#             bboxes[counter] = np.array([u_min, u_max, v_min, v_max])
#
#             class_labels[counter] = class_string_to_label[label_2d["class"]]
#
#             counter += 1
#
#     bboxes = bboxes[0:counter]
#     class_labels = class_labels[0:counter]
#
#     example["bboxes"] = bboxes
#     example["class_labels"] = class_labels
#     examples.append(example)
#
# total_count = 0
# total_count_0 = 0
# total_count_1 = 0
# total_count_2 = 0
# total_count_3 = 0
# for example in examples:
#     gt_bboxes = torch.from_numpy(example["bboxes"]) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
#     gt_classes = torch.from_numpy(example["class_labels"]) # (shape (num_gt_objects, ))
#     label_regr, label_class = bbox_encoder.encode(gt_bboxes, gt_classes)
#     # (label_class is a Tensor of shape: (num_anchors, ))
#
#     label_class = label_class.data.numpy()
#
#     count_0 = np.sum(np.equal(label_class, 0))
#     count_1 = np.sum(np.equal(label_class, 1))
#     count_2 = np.sum(np.equal(label_class, 2))
#     count_3 = np.sum(np.equal(label_class, 3))
#
#     total_count_0 += count_0
#     total_count_1 += count_1
#     total_count_2 += count_2
#     total_count_3 += count_3
#
#     total_count += count_0 + count_1 + count_2 + count_3
#
# print (total_count)
# print (total_count_0)
# print (total_count_1)
# print (total_count_2)
# print (total_count_3)
#
# # compute the class weights according to the ENet paper:
# prob_0 = float(total_count_0)/float(total_count)
# prob_1 = float(total_count_1)/float(total_count)
# prob_2 = float(total_count_2)/float(total_count)
# prob_3 = float(total_count_3)/float(total_count)
#
# weight_0 = 1/np.log(1.02 + prob_0)
# weight_1 = 1/np.log(1.02 + prob_1)
# weight_2 = 1/np.log(1.02 + prob_2)
# weight_3 = 1/np.log(1.02 + prob_3)
#
# class_weights = [weight_0, weight_1, weight_2, weight_3]
#
# print (class_weights)
#
# with open(kitti_meta_path + "/class_weights.pkl", "wb") as file:
#     pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
################################################################################














from kittiloader import LabelLoader2D3D # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import pickle
import numpy as np
import cv2
import math

from datasets import BboxEncoder, class_string_to_label

################################################################################
# compute the class weigths for cars, pedestrians and cyclists:
################################################################################
# (0: background, 1: car, 2: pedestrian, 3: cyclist)

kitti_data_path = "/root/3DOD_thesis/data/kitti"
kitti_meta_path = "/root/retinanet/data/kitti/meta"

img_dir = kitti_data_path + "/object/training/image_2/"
label_dir = kitti_data_path + "/object/training/label_2/"
calib_dir = kitti_data_path + "/object/training/calib/"

with open(kitti_meta_path + "/train_img_ids.pkl", "rb") as file: # (needed for python3)
    train_img_ids = pickle.load(file)

img_height = 375
img_width = 1242

bbox_encoder = BboxEncoder(img_h=img_height, img_w=img_width)

examples = []
for img_id in train_img_ids:
    example = {}
    example["img_id"] = img_id

    labels = LabelLoader2D3D(img_id, label_dir, ".txt", calib_dir, ".txt")

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
    examples.append(example)

total_count = 0
total_count_1 = 0
total_count_2 = 0
total_count_3 = 0
for example in examples:
    gt_bboxes = torch.from_numpy(example["bboxes"]) # (shape: (num_gt_objects, 4), (x_min, x_max, y_min, y_max))
    gt_classes = torch.from_numpy(example["class_labels"]) # (shape (num_gt_objects, ))
    label_regr, label_class = bbox_encoder.encode(gt_bboxes, gt_classes)
    # (label_class is a Tensor of shape: (num_anchors, ))

    label_class = label_class.data.numpy()

    count_1 = np.sum(np.equal(label_class, 1))
    count_2 = np.sum(np.equal(label_class, 2))
    count_3 = np.sum(np.equal(label_class, 3))

    total_count_1 += count_1
    total_count_2 += count_2
    total_count_3 += count_3

    total_count += count_1 + count_2 + count_3

print (total_count)
print (total_count_1)
print (total_count_2)
print (total_count_3)

# compute the class weights according to the ENet paper:
prob_1 = float(total_count_1)/float(total_count)
prob_2 = float(total_count_2)/float(total_count)
prob_3 = float(total_count_3)/float(total_count)

weight_1 = 1/np.log(1.02 + prob_1)
weight_2 = 1/np.log(1.02 + prob_2)
weight_3 = 1/np.log(1.02 + prob_3)

class_weights = [1.0, weight_1, weight_2, weight_3] # (class_weights needs to be of length 4, but the value for background (1.0) won't actually be used)

print (class_weights)

with open(kitti_meta_path + "/class_weights.pkl", "wb") as file:
    pickle.dump(class_weights, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
