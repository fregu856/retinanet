# camera-ready

import pickle
import numpy as np
import os
import math

import torch

from datasets import BboxEncoder

experiment_name = "val_model_8_2_epoch_150" # NOTE change this for every new experiment

# NOTE! here you can choose what model's output you want to compute metrics for
with open("/home/fregu856/retinanet/training_logs/model_8_2/eval_dict_val_150.pkl", "rb") as file: # NOTE! you'll have to adapt this for your file structure
    eval_dict = pickle.load(file)

class_label_to_string = {1: "Car",
                         2: "Pedestrian",
                         3: "Cyclist"} # (background: 0)

img_height = 375
img_width = 1242
bbox_encoder = BboxEncoder(img_h=img_height, img_w=img_width)

project_dir = "/home/fregu856/retinanet/" # NOTE! you'll have to adapt this for your file structure
data_dir = project_dir + "data/kitti/object/training/"
calib_dir = data_dir + "calib/"

eval_kitti_dir = project_dir + "/eval_kitti/"
results_dir = eval_kitti_dir + "build/results/"

experiment_results_dir = results_dir + experiment_name + "/"
results_data_dir = experiment_results_dir + "data/"
if os.path.exists(experiment_results_dir):
    raise Exception("That experiment name already exists!")
else:
    os.makedirs(experiment_results_dir)
    os.makedirs(results_data_dir)

training_img_ids = []
img_names = os.listdir(calib_dir)
for img_name in img_names:
    img_id = img_name.split(".txt")[0]
    training_img_ids.append(img_id)

for img_id in training_img_ids:
    print (img_id)

    img_label_file_path = results_data_dir + img_id + ".txt"
    with open(img_label_file_path, "w") as img_label_file:

        if img_id in eval_dict: # (if any predicted bboxes for the image:) (otherwise, just create an empty file)
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

                for i in range(pred_bboxes_xxyy.shape[0]):
                    pred_bbox_xxyy = pred_bboxes_xxyy[i]
                    pred_prob = pred_probs[i]
                    pred_class_label = pred_class_labels[i]

                    left = pred_bbox_xxyy[0]
                    top = pred_bbox_xxyy[2]
                    right = pred_bbox_xxyy[1]
                    bottom = pred_bbox_xxyy[3]

                    pred_class_string = class_label_to_string[pred_class_label]

                    score = pred_prob

                    # (type, truncated, occluded, alpha, left, top, right, bottom, h, w, l, x, y, z, ry, score)
                    img_label_file.write("%s -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.2f\n" % (pred_class_string, left, top, right, bottom, score))
