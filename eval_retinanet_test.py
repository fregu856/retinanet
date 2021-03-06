from datasets import DatasetTest, BboxEncoder # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from retinanet import RetinaNet

from utils import onehot_embed

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

batch_size = 16

network = RetinaNet("eval_test", project_dir="/root/retinanet").cuda()
network.load_state_dict(torch.load("/root/retinanet/training_logs/model_9_2_2_2/checkpoints/model_9_2_2_2_epoch_390.pth"))

num_classes = network.num_classes

test_dataset = DatasetTest(kitti_data_path="/root/3DOD_thesis/data/kitti",
                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta")

bbox_encoder = BboxEncoder(img_h=test_dataset.img_height, img_w=test_dataset.img_width)

num_test_batches = int(len(test_dataset)/batch_size)
print ("num_test_batches:", num_test_batches)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, shuffle=False,
                                          num_workers=4)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
eval_dict = {}
for step, (imgs, img_ids) in enumerate(test_loader):
    with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))

        outputs = network(imgs)
        outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))

        ########################################################################
        # save data for visualization:
        ########################################################################
        for i in range(outputs_regr.size()[0]):
            img_id = img_ids[i]

            output_regr = outputs_regr[i, :, :].data.cpu() # (shape: (num_anchors, 4))
            output_class = outputs_class[i, :, :].data.cpu() # (num_classes, num_anchors)

            output_class = torch.t(output_class) # (num_anchors, num_classes)

            pred_bboxes, pred_max_scores, pred_class_labels = bbox_encoder.decode(output_regr, output_class)
            # (pred_bboxes has shape (num_preds_after_nms, 4), (x, y, w, h))
            # (pred_max_scores has shape (num_preds_after_nms, ))
            # (pred_class_labels has shape (num_preds_after_nms, ))

            if pred_bboxes is not None:
                pred_bboxes = pred_bboxes.data.cpu().numpy()
                pred_max_scores = pred_max_scores.data.cpu().numpy()
                pred_class_labels = pred_class_labels.data.cpu().numpy()

                img_dict = {}
                img_dict["pred_bboxes"] = pred_bboxes
                img_dict["pred_max_scores"] = pred_max_scores
                img_dict["pred_class_labels"] = pred_class_labels

                eval_dict[img_id] = img_dict

with open("%s/eval_dict_test.pkl" % network.model_dir, "wb") as file:
    pickle.dump(eval_dict, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
