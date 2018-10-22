# TODO! set the loss to the exact same one that I eventually settle on

from datasets import DatasetEvalSeq, BboxEncoder # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
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

lambda_value = 100.0 # (loss weight)
lambda_value_neg = 1.0

network = RetinaNet("eval_val_seq", project_dir="/root/retinanet").cuda()
network.load_state_dict(torch.load("/root/retinanet/training_logs/model_9_2_2_2/checkpoints/model_9_2_2_2_epoch_390.pth"))

num_classes = network.num_classes

placeholder_dataset = DatasetEvalSeq(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                sequence="0000")

bbox_encoder = BboxEncoder(img_h=placeholder_dataset.img_height, img_w=placeholder_dataset.img_width)

regression_loss_func = nn.SmoothL1Loss()
classification_loss_func = nn.CrossEntropyLoss(ignore_index=-1)

network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)

for sequence in ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020"]:
    print (sequence)

    val_dataset = DatasetEvalSeq(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                 kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                 sequence=sequence)

    num_val_batches = int(len(val_dataset)/batch_size)
    print ("num_val_batches:", num_val_batches)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

    batch_losses = []
    batch_losses_class = []
    batch_losses_regr = []
    eval_dict = {}
    for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
            labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
            labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size, num_anchors))

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

                    label_regr = labels_regr[i, :, :].data.cpu() # (shape: (num_anchors, 4))
                    label_class = labels_class[i, :].data.cpu().numpy() # (num_anchors, )

                    gt_bboxes = bbox_encoder.decode_gt_single(label_regr) # (shape: (num_anchors, 4))
                    gt_bboxes = gt_bboxes.numpy()

                    mask = label_class > 0
                    gt_bboxes = gt_bboxes[mask, :] # (shape: (num_gt_objects, 4))
                    gt_class_labels = label_class[mask] # (shape: (num_gt_objects, ))

                    img_dict = {}
                    img_dict["pred_bboxes"] = pred_bboxes
                    img_dict["pred_max_scores"] = pred_max_scores
                    img_dict["pred_class_labels"] = pred_class_labels
                    img_dict["gt_bboxes"] = gt_bboxes
                    img_dict["gt_class_labels"] = gt_class_labels

                    eval_dict[img_id] = img_dict
                else:
                    label_regr = labels_regr[i, :, :].data.cpu() # (shape: (num_anchors, 4))
                    label_class = labels_class[i, :].data.cpu().numpy() # (num_anchors, )

                    gt_bboxes = bbox_encoder.decode_gt_single(label_regr) # (shape: (num_anchors, 4))
                    gt_bboxes = gt_bboxes.numpy()

                    mask = label_class > 0
                    gt_bboxes = gt_bboxes[mask, :] # (shape: (num_gt_objects, 4))
                    gt_class_labels = label_class[mask] # (shape: (num_gt_objects, ))

                    img_dict = {}
                    img_dict["pred_bboxes"] = None
                    img_dict["pred_max_scores"] = None
                    img_dict["pred_class_labels"] = None
                    img_dict["gt_bboxes"] = gt_bboxes
                    img_dict["gt_class_labels"] = gt_class_labels

                    eval_dict[img_id] = img_dict

            ########################################################################
            # # compute the regression loss:
            ########################################################################
            # remove entries which should be ignored (-1 and 0):
            mask = labels_class > 0 # (shape: (batch_size, num_anchors), entries to be ignored are 0, the rest are 1)
            mask = mask.unsqueeze(-1).expand_as(outputs_regr) # (shape: (batch_size, num_anchors, 4))
            mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
            outputs_regr = outputs_regr[mask] # (shape: (num_regr_anchors_in_batch*4, ))
            labels_regr = labels_regr[mask] # (shape: (num_regr_anchors_in_batch*4, ))

            loss_regr = regression_loss_func(outputs_regr, labels_regr)

            loss_regr_value = loss_regr.data.cpu().numpy()
            batch_losses_regr.append(loss_regr_value)

            ########################################################################
            # # compute the classification loss:
            ########################################################################
            # (outputs_class has shape: (batch_size, num_classes, num_anchors))

            labels_class_background = labels_class.clone() # (shape: (batch_size, num_anchors))
            labels_class_background[labels_class_background > 0] = -1 # (shape: (batch_size, num_anchors))
            loss_class_background = classification_loss_func(outputs_class, labels_class_background)

            labels_class_foreground = labels_class.clone() # (shape: (batch_size, num_anchors))
            labels_class_foreground[labels_class_foreground == 0] = -1 # (shape: (batch_size, num_anchors))
            loss_class_foreground = classification_loss_func(outputs_class, labels_class_foreground)

            loss_class = loss_class_foreground + lambda_value_neg*loss_class_background

            loss_class_value = loss_class.data.cpu().numpy()
            batch_losses_class.append(loss_class_value)

            ########################################################################
            # # compute the total loss:
            ########################################################################
            loss = loss_class + lambda_value*loss_regr

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    print ("val loss: %g" % epoch_loss)

    epoch_loss = np.mean(batch_losses_class)
    print ("val class loss: %g" % epoch_loss)

    epoch_loss = np.mean(batch_losses_regr)
    print ("val regr loss: %g" % epoch_loss)

    with open("%s/eval_dict_val_seq_%s.pkl" % (network.model_dir, sequence), "wb") as file:
        pickle.dump(eval_dict, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
