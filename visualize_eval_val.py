import pickle
import numpy as np
import math
import cv2

import torch

from datasets import BboxEncoder

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

def draw_2d_polys_no_text(img, polys):
    img = np.copy(img)
    for poly in polys:
        if 'color' in poly:
            bg = poly['color']
        else:
            bg = np.array([100, 100, 100], dtype='float64')

        cv2.polylines(img, np.int32([poly['poly']]), True, bg, lineType=cv2.LINE_AA, thickness=2)

    return img

img_height = 375
img_width = 1242

project_dir = "/home/fregu856/exjobb/"
data_dir = project_dir + "data/kitti/object/training/"
img_dir = data_dir + "image_2/"

bbox_encoder = BboxEncoder(img_h=img_height, img_w=img_width)

# NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! NOTE! # NOTE!
with open("/home/fregu856/retinanet/training_logs/model_8_2/eval_dict_train_50.pkl", "rb") as file:
    eval_dict = pickle.load(file)

for img_id in eval_dict:
    if img_id in ["000000", "000001", "000002", "000003", "000004", "000005", "000006", "000007", "000008", "000009", "000010", "000011", "000012", "000013", "000014", "000015", "000016", "000017", "000018", "000019", "000020", "000021", "000022", "000023", "000024", "000025", "000026", "000027", "000028", "000029", "000030"]:
        print img_id

        img_dict = eval_dict[img_id]

        img = cv2.imread(img_dir + img_id + ".png", -1)

        pred_bboxes_xywh = torch.from_numpy(img_dict["pred_bboxes"])
        pred_probs = img_dict["pred_max_scores"]
        pred_class_labels = img_dict["pred_class_labels"]
        pred_bboxes_xxyy = bbox_encoder._xywh_2_xxyy(pred_bboxes_xywh).numpy()

        gt_bboxes_xywh = torch.from_numpy(img_dict["gt_bboxes"])
        gt_class_labels = img_dict["gt_class_labels"]
        gt_bboxes_xxyy = bbox_encoder._xywh_2_xxyy(gt_bboxes_xywh).numpy()

        pred_bbox_polys = []
        for i in range(pred_bboxes_xxyy.shape[0]):
            pred_bbox_xxyy = pred_bboxes_xxyy[i]
            pred_prob = pred_probs[i]
            pred_class_label = pred_class_labels[i]

            pred_bbox_poly = create2Dbbox_poly(pred_bbox_xxyy)

            if pred_class_label == 0: # (Car)
                pred_bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
            elif pred_class_label == 1: # (Pedestrian)
                pred_bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
            elif pred_class_label == 2: # (Cyclist)
                pred_bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')

            pred_bbox_poly["prob"] = pred_prob

            pred_bbox_polys.append(pred_bbox_poly)

        gt_bbox_polys = []
        for i in range(gt_bboxes_xxyy.shape[0]):
            gt_bbox_xxyy = gt_bboxes_xxyy[i]
            gt_class_label = gt_class_labels[i]

            gt_bbox_poly = create2Dbbox_poly(gt_bbox_xxyy)

            if gt_class_label == 0: # (Car)
                gt_bbox_poly["color"] = np.array([255, 0, 0], dtype='float64')
            elif gt_class_label == 1: # (Pedestrian)
                gt_bbox_poly["color"] = np.array([0, 200, 0], dtype='float64')
            elif gt_class_label == 2: # (Cyclist)
                gt_bbox_poly["color"] = np.array([0, 0, 255], dtype='float64')

            gt_bbox_polys.append(gt_bbox_poly)

        img_with_pred_bboxes = draw_2d_polys(img, pred_bbox_polys)
        img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (img_width, img_height))

        img_with_gt_bboxes = draw_2d_polys_no_text(img, gt_bbox_polys)
        img_with_gt_bboxes = cv2.resize(img_with_gt_bboxes, (img_width, img_height))

        combined_img = np.zeros((2*img_height, img_width, 3), dtype=np.uint8)
        combined_img[0:img_height] = img_with_gt_bboxes
        combined_img[img_height:] = img_with_pred_bboxes

        cv2.imshow("test", combined_img)
        cv2.waitKey(0)
