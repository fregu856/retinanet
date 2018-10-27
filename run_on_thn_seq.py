from datasets import DatasetThnSeq, BboxEncoder, DatasetThnSeqSynscapes # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from retinanet import RetinaNet

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
import os

def create2Dbbox_poly(bbox2D):
    u_min = bbox2D[0] # (left)
    u_max = bbox2D[1] # (rigth)
    v_min = bbox2D[2] # (top)
    v_max = bbox2D[3] # (bottom)

    poly = {}
    poly['poly'] = np.array([[u_min, v_min], [u_max, v_min],
                             [u_max, v_max], [u_min, v_max]], dtype='int32')

    return poly

def draw_2d_polys(img, polys):
    img = np.copy(img)
    for poly in polys:
        if 'color' in poly:
            bg = poly['color']
        else:
            bg = np.array([0, 255, 0], dtype='float64')

        cv2.polylines(img, np.int32([poly['poly']]), True, bg, lineType=cv2.LINE_AA, thickness=2)

        text = "%.2f" % poly["prob"]
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        thickness = 1
        t_size, _ = cv2.getTextSize(text, fontFace, fontScale+0.1, thickness)
        p_org = tuple(np.amax(poly['poly'], 0))
        p_top = (p_org[0] - t_size[0], p_org[1] - t_size[1])
        p = (p_top[0] + 1, p_org[1] - 1)
        cv2.rectangle(img, p_top, p_org, bg, cv2.FILLED)
        cv2.putText(img, text, p, fontFace, fontScale, [255, 255, 255], 1, cv2.LINE_AA)

    return img

batch_size = 16

network = RetinaNet("eval_thn", project_dir="/root/retinanet").cuda()
network.load_state_dict(torch.load("/root/retinanet/training_logs/model_13/checkpoints/model_13_epoch_50.pth"))

test_dataset = DatasetThnSeqSynscapes(thn_data_path="/root/deeplabv3/data/thn")
#test_dataset = DatasetThnSeq(thn_data_path="/root/deeplabv3/data/thn")

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

################################################################################
# create visualization video:
################################################################################
orig_img_height = 512
orig_img_width = 1024

img_height = 375
img_width = 1242

img_dir = "/root/deeplabv3/data/thn/"

img_data_dict = {}
for img_id in eval_dict:
    data_dict = {}

    img_dict = eval_dict[img_id]

    if img_dict["pred_bboxes"] is not None:
        pred_bboxes_xywh = torch.from_numpy(img_dict["pred_bboxes"])
        pred_probs = img_dict["pred_max_scores"]
        pred_class_labels = img_dict["pred_class_labels"]
        if pred_bboxes_xywh.size() == torch.Size([4]):
            pred_bboxes_xywh = pred_bboxes_xywh.unsqueeze(0)
            pred_probs = np.array([pred_probs])
            pred_class_labels = np.array([pred_class_labels])
        pred_bboxes_xxyy = bbox_encoder._xywh_2_xxyy(pred_bboxes_xywh).numpy()

        pred_bbox_polys = []
        for i in range(pred_bboxes_xxyy.shape[0]):
            pred_bbox_xxyy = pred_bboxes_xxyy[i]
            pred_prob = pred_probs[i]
            pred_class_label = pred_class_labels[i]

            pred_bbox_poly = create2Dbbox_poly(pred_bbox_xxyy)

            if pred_class_label == 1: # (Car)
                pred_bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
            elif pred_class_label == 2: # (Pedestrian)
                pred_bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
            elif pred_class_label == 3: # (Cyclist)
                pred_bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')

            pred_bbox_poly["prob"] = pred_prob

            pred_bbox_polys.append(pred_bbox_poly)
    else:
        pred_bbox_polys = []

    data_dict["pred_bbox_polys"] = pred_bbox_polys

    img_data_dict[img_id] = data_dict

sorted_img_ids = []
img_names = sorted(os.listdir(img_dir))
for img_name in img_names:
    img_id = img_name.split(".png")[0]
    sorted_img_ids.append(img_id)

out = cv2.VideoWriter("/root/retinanet/training_logs/model_eval_thn/eval_thn_seq_img_pred.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12, (img_width, 2*img_height), True)

for img_id in sorted_img_ids:
    print (img_id)

    img = cv2.imread(img_dir + img_id + ".png", -1)
    img = cv2.resize(img, (img_width, int((img_width/orig_img_width)*img_height)))
    img = img[(int((img_width/orig_img_width)*img_height) - img_height):int((img_width/orig_img_width)*img_height)]

    img_with_pred_bboxes = img

    if img_id in img_data_dict:
        data_dict = img_data_dict[img_id]
        pred_bbox_polys = data_dict["pred_bbox_polys"]

        img_with_pred_bboxes = draw_2d_polys(img_with_pred_bboxes, pred_bbox_polys)

    img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (img_width, img_height))

    combined_img = np.zeros((2*img_height, img_width, 3), dtype=np.uint8)
    combined_img[0:img_height] = cv2.resize(img, (img_width, img_height))
    combined_img[img_height:] = img_with_pred_bboxes

    out.write(combined_img)
