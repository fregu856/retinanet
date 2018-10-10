import pickle
import numpy as np
import math
import cv2
import os

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
#
# sequence = "0000" # NOTE! change this to visualize a different sequence
for sequence in ["0000", "0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017", "0018", "0019", "0020", "0021", "0022", "0023", "0024", "0025", "0026", "0027"]:
    print (sequence)

    # NOTE! here you can choose what model's output you want to visualize
    with open("/root/retinanet/training_logs/model_eval_test_seq/eval_dict_test_seq_%s.pkl" % sequence, "rb") as file:
        eval_dict = pickle.load(file)

    img_height = 375
    img_width = 1242

    project_dir = "/root/3DOD_thesis/"
    data_dir = project_dir + "data/kitti/tracking/testing/"
    img_dir = data_dir + "image_02/" + sequence + "/"

    bbox_encoder = BboxEncoder(img_h=img_height, img_w=img_width)

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

    # ################################################################################
    # # create a video of images (no bboxes):
    # ################################################################################
    # out = cv2.VideoWriter("/root/retinanet/training_logs/model_eval_test_seq/eval_test_seq_%s_img.avi" % sequence, cv2.VideoWriter_fourcc(*"MJPG"), 12, (img_width, img_height), True)
    #
    # for img_id in sorted_img_ids:
    #     print (img_id)
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #
    #     img = cv2.resize(img, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #
    #     out.write(img)

    # ################################################################################
    # # create a video of images with pred:
    # ################################################################################
    # out = cv2.VideoWriter("/root/retinanet/training_logs/model_eval_test_seq/eval_test_seq_%s_img_pred.avi" % sequence, cv2.VideoWriter_fourcc(*"MJPG"), 12, (img_width, img_height), True)
    #
    # for img_id in sorted_img_ids:
    #     print (img_id)
    #
    #     img = cv2.imread(img_dir + img_id + ".png", -1)
    #
    #     img_with_pred_bboxes = img
    #
    #     if img_id in img_data_dict:
    #         data_dict = img_data_dict[img_id]
    #         pred_bbox_polys = data_dict["pred_bbox_polys"]
    #
    #         img_with_pred_bboxes = draw_2d_polys(img_with_pred_bboxes, pred_bbox_polys)
    #
    #     img_with_pred_bboxes = cv2.resize(img_with_pred_bboxes, (img_width, img_height)) # (the image MUST have the size specified in VideoWriter)
    #
    #     out.write(img_with_pred_bboxes)

    ################################################################################
    # create a video of images with just image on top of pred:
    ################################################################################
    out = cv2.VideoWriter("/root/retinanet/training_logs/model_eval_test_seq/eval_test_seq_%s_img_pred.avi" % sequence, cv2.VideoWriter_fourcc(*"MJPG"), 12, (img_width, 2*img_height), True)

    for img_id in sorted_img_ids:
        print (img_id)

        img = cv2.imread(img_dir + img_id + ".png", -1)

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
