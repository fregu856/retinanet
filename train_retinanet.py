# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "1"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.001
# max_grad_norm = 1.0
#
# lambda_value = 100.0 # (loss weight)
# gamma = 4.0
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
# init_weights(network)
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
#
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
#
# #####################
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# #optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
# ####
# # params = add_weight_decay(network, l2_value=0.0001)
# # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
# #
# # #optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
# #####################
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     if epoch % 100 == 0 and epoch > 0:
#         learning_rate = learning_rate/2
#
#         #######################
#         optimizer = torch.optim.Adam(params, lr=learning_rate) # (with weight decay)
#
#         #optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
#         ####
#         # optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
#         #
#         # #optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
#         #######################
#
#     print ("learning_rate:")
#     print (learning_rate)
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#         # NOTE! NOTE! why do I do this? Is this what messes things up? Guess I do this to be able to mask away those that should be removed?
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss (Focal loss):
#         ########################################################################
#         # remove entries which should be ignored (-1):
#         mask = labels_class > -1 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class = outputs_class[mask, :] # (shape: (num_class_anchors, num_classes))
#         labels_class = labels_class[mask] # (shape: (num_class_anchors, ))
#
#         #loss_class = F.nll_loss(F.log_softmax(outputs_class, dim=1), labels_class)
#
#         labels_class_onehot = onehot_embed(labels_class, num_classes) # (shape: (num_class_anchors, num_classes))
#         CE = -labels_class_onehot*F.log_softmax(outputs_class, dim=1) # (shape: (num_class_anchors, num_classes))
#         weight = labels_class_onehot*torch.pow(1 - F.softmax(outputs_class, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#         loss_class = weight*CE # (shape: (num_class_anchors, num_classes))
#         if step == 0:
#             print ("loss_class.data.cpu().numpy():")
#             print (loss_class.data.cpu().numpy())
#             print (loss_class.data.cpu().numpy().shape)
#             print ("loss_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (loss_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (loss_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#         loss_class, _ = torch.max(loss_class, dim=1) # (shape: (num_class_anchors, ))
#         loss_class = torch.sum(loss_class/num_foreground_anchors)
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         torch.nn.utils.clip_grad_norm_(network.parameters(), max_grad_norm) # (clip gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             ########################################################################
#             # # compute the classification loss (Focal loss):
#             ########################################################################
#             # remove entries which should be ignored (-1):
#             mask = labels_class > -1 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class = outputs_class[mask, :] # (shape: (num_class_anchors, num_classes))
#             labels_class = labels_class[mask] # (shape: (num_class_anchors, ))
#
#             #loss_class = F.nll_loss(F.log_softmax(outputs_class, dim=1), labels_class)
#
#             labels_class_onehot = onehot_embed(labels_class, num_classes) # (shape: (num_class_anchors, num_classes))
#
#             CE = -labels_class_onehot*F.log_softmax(outputs_class, dim=1) # (shape: (num_class_anchors, num_classes))
#
#             weight = labels_class_onehot*torch.pow(1 - F.softmax(outputs_class, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#
#             loss_class = weight*CE # (shape: (num_class_anchors, num_classes))
#             loss_class, _ = torch.max(loss_class, dim=1) # (shape: (num_class_anchors, ))
#             loss_class = torch.sum(loss_class/num_foreground_anchors)
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# ################################################################################
# # with normal classification loss instead of focal loss:
# ################################################################################
# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "2"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 10.0 # (loss weight)
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
# classification_loss_func = nn.CrossEntropyLoss()
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#         # NOTE! NOTE! why do I do this? Is this what messes things up? Guess I do this to be able to mask away those that should be removed?
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         # remove entries which should be ignored (-1):
#         mask = labels_class > -1 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class = outputs_class[mask, :] # (shape: (num_class_anchors, num_classes))
#         labels_class = labels_class[mask] # (shape: (num_class_anchors, ))
#
#         loss_class = classification_loss_func(outputs_class, labels_class)
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             # remove entries which should be ignored (-1):
#             mask = labels_class > -1 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class = outputs_class[mask, :] # (shape: (num_class_anchors, num_classes))
#             labels_class = labels_class[mask] # (shape: (num_class_anchors, ))
#
#             loss_class = classification_loss_func(outputs_class, labels_class)
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             if step == 0:
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#                 print (outputs_class.data.cpu().numpy().shape)
#                 print ("labels_class.data.cpu().numpy():")
#                 print (labels_class.data.cpu().numpy())
#                 print (labels_class.data.cpu().numpy().shape)
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# ################################################################################
# # with normal classification loss instead of focal loss AND class weights:
# ################################################################################
# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "3"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 10.0 # (loss weight)
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
#
# with open("/root/retinanet/data/kitti/meta/class_weights.pkl", "rb") as file: # (needed for python3)
#     class_weights = np.array(pickle.load(file))
# class_weights = torch.from_numpy(class_weights)
# class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()
# classification_loss_func = nn.CrossEntropyLoss(weight=class_weights)
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#         # NOTE! NOTE! why do I do this? Is this what messes things up? Guess I do this to be able to mask away those that should be removed?
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         # remove entries which should be ignored (-1):
#         mask = labels_class > -1 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class = outputs_class[mask, :] # (shape: (num_class_anchors, num_classes))
#         labels_class = labels_class[mask] # (shape: (num_class_anchors, ))
#
#         loss_class = classification_loss_func(outputs_class, labels_class)
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             # remove entries which should be ignored (-1):
#             mask = labels_class > -1 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class = outputs_class[mask, :] # (shape: (num_class_anchors, num_classes))
#             labels_class = labels_class[mask] # (shape: (num_class_anchors, ))
#
#             loss_class = classification_loss_func(outputs_class, labels_class)
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             if step == 0:
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#                 print (outputs_class.data.cpu().numpy().shape)
#                 print ("labels_class.data.cpu().numpy():")
#                 print (labels_class.data.cpu().numpy())
#                 print (labels_class.data.cpu().numpy().shape)
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# ################################################################################
# # with normal classification loss instead of focal loss AND separate class. losses for background and foreground:
# ################################################################################
# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "4"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 10.0 # (loss weight)
# lambda_value_neg = 1.0
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
# classification_loss_func = nn.CrossEntropyLoss()
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#         # NOTE! NOTE! why do I do this? Is this what messes things up? Guess I do this to be able to mask away those that should be removed?
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         mask_background = labels_class == 0
#         mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#         mask_foreground = labels_class > 0
#         mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_foreground = labels_class[mask_foreground] # (shape: (num_background_anchors, ))
#
#         num_foreground_anchors = float(labels_class_foreground.size()[0])
#         num_background_anchors = float(labels_class_background.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#             print ("num_background_anchors:")
#             print (num_background_anchors)
#
#         loss_class_foreground = classification_loss_func(outputs_class_foreground, labels_class_foreground)
#
#         loss_class_background = classification_loss_func(outputs_class_background, labels_class_background)
#
#         loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             mask_background = labels_class == 0
#             mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#             mask_foreground = labels_class > 0
#             mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_foreground = labels_class[mask_foreground] # (shape: (num_background_anchors, ))
#
#             num_foreground_anchors = float(labels_class_foreground.size()[0])
#             num_background_anchors = float(labels_class_background.size()[0])
#
#             if step == 0:
#                 print ("num_foreground_anchors:")
#                 print (num_foreground_anchors)
#                 print ("num_background_anchors:")
#                 print (num_background_anchors)
#
#             loss_class_foreground = classification_loss_func(outputs_class_foreground, labels_class_foreground)
#
#             loss_class_background = classification_loss_func(outputs_class_background, labels_class_background)
#
#             loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             if step == 0:
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#                 print (outputs_class.data.cpu().numpy().shape)
#                 print ("labels_class.data.cpu().numpy():")
#                 print (labels_class.data.cpu().numpy())
#                 print (labels_class.data.cpu().numpy().shape)
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# ################################################################################
# # with normal classification loss instead of focal loss AND separate class. losses for background and foreground AND hard negative mining:
# ################################################################################
# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "5"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 10.0 # (loss weight)
# lambda_value_neg = 1.0
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
# classification_loss_func = nn.CrossEntropyLoss()
# classification_loss_func_background = nn.CrossEntropyLoss(reduce=False)
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#         # NOTE! NOTE! why do I do this? Is this what messes things up? Guess I do this to be able to mask away those that should be removed?
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         mask_background = labels_class == 0
#         mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#         mask_foreground = labels_class > 0
#         mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_foreground = labels_class[mask_foreground] # (shape: (num_background_anchors, ))
#
#         num_foreground_anchors = float(labels_class_foreground.size()[0])
#         num_background_anchors = float(labels_class_background.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#             print ("num_background_anchors:")
#             print (num_background_anchors)
#
#         loss_class_foreground = classification_loss_func(outputs_class_foreground, labels_class_foreground)
#
#
#
#
#
#         loss_class_background = classification_loss_func_background(outputs_class_background, labels_class_background) # (shape: (num_background_anchors, ))
#         if step == 0:
#             print (loss_class_background)
#             print (loss_class_background.size())
#
#         mask_class_background = loss_class_background > 1.0
#         mask_class_background = mask_class_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         loss_class_background = loss_class_background[mask_class_background]
#         if step == 0:
#             print (loss_class_background)
#             print (loss_class_background.size())
#         loss_class_background = torch.mean(loss_class_background)
#
#
#
#
#         loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             mask_background = labels_class == 0
#             mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#             mask_foreground = labels_class > 0
#             mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_foreground = labels_class[mask_foreground] # (shape: (num_background_anchors, ))
#
#             num_foreground_anchors = float(labels_class_foreground.size()[0])
#             num_background_anchors = float(labels_class_background.size()[0])
#
#             if step == 0:
#                 print ("num_foreground_anchors:")
#                 print (num_foreground_anchors)
#                 print ("num_background_anchors:")
#                 print (num_background_anchors)
#
#             loss_class_foreground = classification_loss_func(outputs_class_foreground, labels_class_foreground)
#
#
#
#
#
#             loss_class_background = classification_loss_func_background(outputs_class_background, labels_class_background) # (shape: (num_background_anchors, ))
#             if step == 0:
#                 print (loss_class_background)
#                 print (loss_class_background.size())
#
#             mask_class_background = loss_class_background > 1.0
#             mask_class_background = mask_class_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             loss_class_background = loss_class_background[mask_class_background]
#             if step == 0:
#                 print (loss_class_background)
#                 print (loss_class_background.size())
#             loss_class_background = torch.mean(loss_class_background)
#
#
#
#
#             loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             if step == 0:
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#                 print (outputs_class.data.cpu().numpy().shape)
#                 print ("labels_class.data.cpu().numpy():")
#                 print (labels_class.data.cpu().numpy())
#                 print (labels_class.data.cpu().numpy().shape)
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# ################################################################################
# # with focal loss AND separate class. losses for background and foreground:
# ################################################################################
# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "6_2"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 10.0 # (loss weight)
# lambda_value_neg = 1.0
#
# gamma = 2.0
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
# classification_loss_func = nn.CrossEntropyLoss()
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     if epoch % 25 == 0 and epoch > 0:
#         learning_rate = learning_rate/2
#         optimizer = torch.optim.Adam(params, lr=learning_rate)
#
#     print ("learning_rate:")
#     print (learning_rate)
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#         # NOTE! NOTE! why do I do this? Is this what messes things up? Guess I do this to be able to mask away those that should be removed?
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         mask_background = labels_class == 0
#         mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#         mask_foreground = labels_class > 0
#         mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_foreground = labels_class[mask_foreground] # (shape: (num_background_anchors, ))
#
#         num_foreground_anchors = float(labels_class_foreground.size()[0])
#         num_background_anchors = float(labels_class_background.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#             print ("num_background_anchors:")
#             print (num_background_anchors)
#
#         labels_class_onehot_foreground = onehot_embed(labels_class_foreground, num_classes) # (shape: (num_class_anchors, num_classes))
#         CE_foreground = -labels_class_onehot_foreground*F.log_softmax(outputs_class_foreground, dim=1) # (shape: (num_class_anchors, num_classes))
#         weight_foreground = labels_class_onehot_foreground*torch.pow(1 - F.softmax(outputs_class_foreground, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#         loss_class_foreground = weight_foreground*CE_foreground # (shape: (num_class_anchors, num_classes))
#         loss_class_foreground, _ = torch.max(loss_class_foreground, dim=1) # (shape: (num_class_anchors, ))
#         loss_class_foreground = torch.mean(loss_class_foreground)
#
#         labels_class_onehot_background = onehot_embed(labels_class_background, num_classes) # (shape: (num_class_anchors, num_classes))
#         CE_background = -labels_class_onehot_background*F.log_softmax(outputs_class_background, dim=1) # (shape: (num_class_anchors, num_classes))
#         weight_background = labels_class_onehot_background*torch.pow(1 - F.softmax(outputs_class_background, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#         loss_class_background = weight_background*CE_background # (shape: (num_class_anchors, num_classes))
#         loss_class_background, _ = torch.max(loss_class_background, dim=1) # (shape: (num_class_anchors, ))
#         loss_class_background = torch.mean(loss_class_background)
#
#         loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             outputs_regr = outputs_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             mask_background = labels_class == 0
#             mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#             mask_foreground = labels_class > 0
#             mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_foreground = labels_class[mask_foreground] # (shape: (num_background_anchors, ))
#
#             num_foreground_anchors = float(labels_class_foreground.size()[0])
#             num_background_anchors = float(labels_class_background.size()[0])
#
#             if step == 0:
#                 print ("num_foreground_anchors:")
#                 print (num_foreground_anchors)
#                 print ("num_background_anchors:")
#                 print (num_background_anchors)
#
#             labels_class_onehot_foreground = onehot_embed(labels_class_foreground, num_classes) # (shape: (num_class_anchors, num_classes))
#             CE_foreground = -labels_class_onehot_foreground*F.log_softmax(outputs_class_foreground, dim=1) # (shape: (num_class_anchors, num_classes))
#             weight_foreground = labels_class_onehot_foreground*torch.pow(1 - F.softmax(outputs_class_foreground, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#             loss_class_foreground = weight_foreground*CE_foreground # (shape: (num_class_anchors, num_classes))
#             loss_class_foreground, _ = torch.max(loss_class_foreground, dim=1) # (shape: (num_class_anchors, ))
#             loss_class_foreground = torch.mean(loss_class_foreground)
#
#             labels_class_onehot_background = onehot_embed(labels_class_background, num_classes) # (shape: (num_class_anchors, num_classes))
#             CE_background = -labels_class_onehot_background*F.log_softmax(outputs_class_background, dim=1) # (shape: (num_class_anchors, num_classes))
#             weight_background = labels_class_onehot_background*torch.pow(1 - F.softmax(outputs_class_background, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#             loss_class_background = weight_background*CE_background # (shape: (num_class_anchors, num_classes))
#             loss_class_background, _ = torch.max(loss_class_background, dim=1) # (shape: (num_class_anchors, ))
#             loss_class_background = torch.mean(loss_class_background)
#
#             loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             if step == 0:
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#                 print (outputs_class.data.cpu().numpy().shape)
#                 print ("labels_class.data.cpu().numpy():")
#                 print (labels_class.data.cpu().numpy())
#                 print (labels_class.data.cpu().numpy().shape)
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

# ################################################################################
# # with normal class. loss AND separate class. losses for background and foreground AND attempt at not using .view():
# ################################################################################
# from datasets import DatasetAugmentation, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "7_3"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 100.0 # (loss weight)
# lambda_value_neg = 1.0
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetAugmentation(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
# classification_loss_func = nn.CrossEntropyLoss(ignore_index=-1)
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     if epoch % 25 == 0 and epoch > 0:
#         learning_rate = learning_rate/2
#         optimizer = torch.optim.Adam(params, lr=learning_rate)
#
#     print ("learning_rate:")
#     print (learning_rate)
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should always be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size, num_anchors), entries to be ignored are 0, the rest are 1)
#         mask = mask.unsqueeze(-1).expand_as(outputs_regr) # (shape: (batch_size, num_anchors, 4))
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask] # (shape: (num_regr_anchors_in_batch*4, ))
#         labels_regr = labels_regr[mask] # (shape: (num_regr_anchors_in_batch*4, ))
#
#         num_foreground_anchors = float(labels_regr.size()[0]/4)
#         #
#         # if step == 0:
#         #     print ("num_foreground_anchors:")
#         #     print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         # if step == 0:
#         #     print ("outputs_regr.data.cpu().numpy():")
#         #     print (outputs_regr.data.cpu().numpy())
#         #     print (outputs_regr.data.cpu().numpy().shape)
#         #     print ("labels_regr.data.cpu().numpy():")
#         #     print (labels_regr.data.cpu().numpy())
#         #     print (labels_regr.data.cpu().numpy().shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         # (outputs_class has shape: (batch_size, num_classes, num_anchors))
#
#         labels_class_background = labels_class.clone() # (shape: (batch_size, num_anchors))
#         labels_class_background[labels_class_background > 0] = -1 # (shape: (batch_size, num_anchors))
#         loss_class_background = classification_loss_func(outputs_class, labels_class_background)
#
#         labels_class_foreground = labels_class.clone() # (shape: (batch_size, num_anchors))
#         labels_class_foreground[labels_class_foreground == 0] = -1 # (shape: (batch_size, num_anchors))
#         loss_class_foreground = classification_loss_func(outputs_class, labels_class_foreground)
#
#         num_foreground_anchors = labels_class_foreground.size()[0]*labels_class_foreground.size()[1] - np.sum(np.equal(labels_class_foreground.data.cpu().numpy(), -1))
#         num_background_anchors = labels_class_background.size()[0]*labels_class_background.size()[1] - np.sum(np.equal(labels_class_background.data.cpu().numpy(), -1))
#
#         # if step == 0:
#         #     print ("num_foreground_anchors:")
#         #     print (num_foreground_anchors)
#         #     print ("num_background_anchors:")
#         #     print (num_background_anchors)
#
#         loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         # if step == 0:
#         #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#         #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#         #     print (outputs_class.data.cpu().numpy().shape)
#         #     print ("labels_class.data.cpu().numpy():")
#         #     print (labels_class.data.cpu().numpy())
#         #     print (labels_class.data.cpu().numpy().shape)
#             # print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             # print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             # print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             # print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             # print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             # print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         # if step == 0:
#         #     print ("#################################################")
#         #     print ("################### GRADIENTS ###################")
#         #     print ("#################################################")
#         #     print ("network.class_conv1.weight.grad:")
#         #     print (network.class_conv1.weight.grad)
#         #     print ("network.class_conv2.weight.grad:")
#         #     print (network.class_conv2.weight.grad)
#         #     print ("network.class_conv3.weight.grad:")
#         #     print (network.class_conv3.weight.grad)
#         #     print ("network.class_conv4.weight.grad:")
#         #     print (network.class_conv4.weight.grad)
#         #     print ("network.class_conv5.weight.grad:")
#         #     print (network.class_conv5.weight.grad)
#         #
#         #     print ("network.regr_conv1.weight.grad:")
#         #     print (network.regr_conv1.weight.grad)
#         #     print ("network.regr_conv2.weight.grad:")
#         #     print (network.regr_conv2.weight.grad)
#         #     print ("network.regr_conv3.weight.grad:")
#         #     print (network.regr_conv3.weight.grad)
#         #     print ("network.regr_conv4.weight.grad:")
#         #     print (network.regr_conv4.weight.grad)
#         #     print ("network.regr_conv5.weight.grad:")
#         #     print (network.regr_conv5.weight.grad)
#         #
#         #     print ("network.fpn.conv6.weight.grad:")
#         #     print (network.fpn.conv6.weight.grad)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should always be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size, num_anchors), entries to be ignored are 0, the rest are 1)
#             mask = mask.unsqueeze(-1).expand_as(outputs_regr) # (shape: (batch_size, num_anchors, 4))
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask] # (shape: (num_regr_anchors_in_batch*4, ))
#             labels_regr = labels_regr[mask] # (shape: (num_regr_anchors_in_batch*4, ))
#
#             num_foreground_anchors = float(labels_regr.size()[0]/4)
#             #
#             # if step == 0:
#             #     print ("num_foreground_anchors:")
#             #     print (num_foreground_anchors)
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             # if step == 0:
#             #     print ("outputs_regr.data.cpu().numpy():")
#             #     print (outputs_regr.data.cpu().numpy())
#             #     print (outputs_regr.data.cpu().numpy().shape)
#             #     print ("labels_regr.data.cpu().numpy():")
#             #     print (labels_regr.data.cpu().numpy())
#             #     print (labels_regr.data.cpu().numpy().shape)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             # (outputs_class has shape: (batch_size, num_classes, num_anchors))
#
#             labels_class_background = labels_class.clone() # (shape: (batch_size, num_anchors))
#             labels_class_background[labels_class_background > 0] = -1 # (shape: (batch_size, num_anchors))
#             loss_class_background = classification_loss_func(outputs_class, labels_class_background)
#
#             labels_class_foreground = labels_class.clone() # (shape: (batch_size, num_anchors))
#             labels_class_foreground[labels_class_foreground == 0] = -1 # (shape: (batch_size, num_anchors))
#             loss_class_foreground = classification_loss_func(outputs_class, labels_class_foreground)
#
#             num_foreground_anchors = labels_class_foreground.size()[0]*labels_class_foreground.size()[1] - np.sum(np.equal(labels_class_foreground.data.cpu().numpy(), -1))
#             num_background_anchors = labels_class_background.size()[0]*labels_class_background.size()[1] - np.sum(np.equal(labels_class_background.data.cpu().numpy(), -1))
#
#             # if step == 0:
#             #     print ("num_foreground_anchors:")
#             #     print (num_foreground_anchors)
#             #     print ("num_background_anchors:")
#             #     print (num_background_anchors)
#
#             loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             # if step == 0:
#             #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             #     print (outputs_class.data.cpu().numpy().shape)
#             #     print ("labels_class.data.cpu().numpy():")
#             #     print (labels_class.data.cpu().numpy())
#             #     print (labels_class.data.cpu().numpy().shape)
#                 # print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 # print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 # print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 # print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 # print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 # print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

################################################################################
# with focal loss AND separate class. losses for background and foreground AND with the contiguous fix:
################################################################################
# from datasets import DatasetMoreAugmentation2, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
# from retinanet import RetinaNet
#
# from utils import onehot_embed, init_weights, add_weight_decay
#
# import torch
# import torch.utils.data
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.optim as optim
# import torch.nn.functional as F
#
# import numpy as np
# import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import cv2
#
# import time
#
# # NOTE! change this to not overwrite all log data when you train the model:
# model_id = "9_2_3_2"
#
# num_epochs = 1000
# batch_size = 16
# learning_rate = 0.0001
#
# lambda_value = 100.0 # (loss weight)
# lambda_value_neg = 80.0
#
# gamma = 2.0
#
# network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()
#
# num_classes = network.num_classes
#
# train_dataset = DatasetMoreAugmentation2(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                                     kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                                     type="train")
# val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
#                           kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
#                           type="val")
#
# num_train_batches = int(len(train_dataset)/batch_size)
# num_val_batches = int(len(val_dataset)/batch_size)
# print ("num_train_batches:", num_train_batches)
# print ("num_val_batches:", num_val_batches)
#
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size, shuffle=True,
#                                            num_workers=4)
# val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                          batch_size=batch_size, shuffle=False,
#                                          num_workers=4)
#
# regression_loss_func = nn.SmoothL1Loss()
#
# params = add_weight_decay(network, l2_value=0.0001)
# optimizer = torch.optim.Adam(params, lr=learning_rate)
#
# with open("/root/retinanet/data/kitti/meta/class_weights.pkl", "rb") as file: # (needed for python3)
#     class_weights_foreground = np.array(pickle.load(file))
# class_weights_foreground = torch.from_numpy(class_weights_foreground)
# class_weights_foreground = Variable(class_weights_foreground.type(torch.FloatTensor)).cuda()
# print (class_weights_foreground)
#
# epoch_losses_train = []
# epoch_losses_class_train = []
# epoch_losses_regr_train = []
# epoch_losses_val = []
# epoch_losses_class_val = []
# epoch_losses_regr_val = []
# for epoch in range(num_epochs):
#     print ("###########################")
#     print ("######## NEW EPOCH ########")
#     print ("###########################")
#     print ("epoch: %d/%d" % (epoch+1, num_epochs))
#
#     ################################################################################
#     # train:
#     ################################################################################
#     network.train() # (set in training mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
#         #current_time = time.time()
#
#         imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#         labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#         outputs = network(imgs)
#         outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#         outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))
#         outputs_class = outputs_class.permute(0, 2, 1) # (shape: (batch_size, num_anchors, num_classes))
#
#         labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#         labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#         labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#         # NOTE! NOTE! I use contiguous() here!
#         outputs_regr = outputs_regr.contiguous().view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#         outputs_class = outputs_class.contiguous().view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#         # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#         # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#         # -1 should be ignored for classification)
#
#         ########################################################################
#         # # compute the regression loss:
#         ########################################################################
#         # remove entries which should be ignored (-1 and 0):
#         mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#         mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#         labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#         num_foreground_anchors = float(labels_regr.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#
#         loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#         loss_regr_value = loss_regr.data.cpu().numpy()
#         batch_losses_regr.append(loss_regr_value)
#
#         if step == 0:
#             print ("outputs_regr.data.cpu().numpy():")
#             print (outputs_regr.data.cpu().numpy())
#             print (outputs_regr.data.cpu().numpy().shape)
#             print ("labels_regr.data.cpu().numpy():")
#             print (labels_regr.data.cpu().numpy())
#             print (labels_regr.data.cpu().numpy().shape)
#
#         # ########################################################################
#         # # # compute the classification loss:
#         # ########################################################################
#         # mask = labels_class > -1
#         # mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         # outputs_class = outputs_class[mask, :] # (shape: (num_anchors, num_classes))
#         # labels_class = labels_class[mask] # (shape: (num_anchors, ))
#         #
#         # labels_class_onehot = onehot_embed(labels_class, num_classes) # (shape: (num_class_anchors, num_classes))
#         # CE = -labels_class_onehot*F.log_softmax(outputs_class, dim=1) # (shape: (num_class_anchors, num_classes))
#         # weight = labels_class_onehot*torch.pow(1 - F.softmax(outputs_class, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#         # loss_class = weight*CE # (shape: (num_class_anchors, num_classes))
#         # loss_class, _ = torch.max(loss_class, dim=1) # (shape: (num_class_anchors, ))
#         # loss_class = torch.mean(loss_class)
#         #
#         # loss_class_value = loss_class.data.cpu().numpy()
#         # batch_losses_class.append(loss_class_value)
#         #
#         # if step == 0:
#         #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#         #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#         #     print (outputs_class.data.cpu().numpy().shape)
#         #     print ("labels_class.data.cpu().numpy():")
#         #     print (labels_class.data.cpu().numpy())
#         #     print (labels_class.data.cpu().numpy().shape)
#         #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#         #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#         #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#         #     print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#         #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#         #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the classification loss:
#         ########################################################################
#         mask_background = labels_class == 0
#         mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#         labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#         mask_foreground = labels_class > 0
#         mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#         outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_foreground_anchors, num_classes))
#         labels_class_foreground = labels_class[mask_foreground] # (shape: (num_foreground_anchors, ))
#
#         num_foreground_anchors = float(labels_class_foreground.size()[0])
#         num_background_anchors = float(labels_class_background.size()[0])
#
#         if step == 0:
#             print ("num_foreground_anchors:")
#             print (num_foreground_anchors)
#             print ("num_background_anchors:")
#             print (num_background_anchors)
#
#         labels_class_onehot_foreground = onehot_embed(labels_class_foreground, num_classes) # (shape: (num_class_anchors, num_classes))
#         CE_foreground = -labels_class_onehot_foreground*F.log_softmax(outputs_class_foreground, dim=1) # (shape: (num_class_anchors, num_classes))
#         weight_foreground = class_weights_foreground*labels_class_onehot_foreground*torch.pow(1 - F.softmax(outputs_class_foreground, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#         loss_class_foreground = weight_foreground*CE_foreground # (shape: (num_class_anchors, num_classes))
#         loss_class_foreground, _ = torch.max(loss_class_foreground, dim=1) # (shape: (num_class_anchors, ))
#         loss_class_foreground = torch.mean(loss_class_foreground)
#
#         labels_class_onehot_background = onehot_embed(labels_class_background, num_classes) # (shape: (num_class_anchors, num_classes))
#         CE_background = -labels_class_onehot_background*F.log_softmax(outputs_class_background, dim=1) # (shape: (num_class_anchors, num_classes))
#         weight_background = labels_class_onehot_background*torch.pow(1 - F.softmax(outputs_class_background, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#         loss_class_background = weight_background*CE_background # (shape: (num_class_anchors, num_classes))
#         loss_class_background, _ = torch.max(loss_class_background, dim=1) # (shape: (num_class_anchors, ))
#         loss_class_background = torch.mean(loss_class_background)
#
#         loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#         loss_class_value = loss_class.data.cpu().numpy()
#         batch_losses_class.append(loss_class_value)
#
#         if step == 0:
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             print (outputs_class.data.cpu().numpy().shape)
#             print ("labels_class.data.cpu().numpy():")
#             print (labels_class.data.cpu().numpy())
#             print (labels_class.data.cpu().numpy().shape)
#             print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#         ########################################################################
#         # # compute the total loss:
#         ########################################################################
#         loss = loss_class + lambda_value*loss_regr
#
#         loss_value = loss.data.cpu().numpy()
#         batch_losses.append(loss_value)
#
#         ########################################################################
#         # optimization step:
#         ########################################################################
#         optimizer.zero_grad() # (reset gradients)
#         loss.backward() # (compute gradients)
#         optimizer.step() # (perform optimization step)
#
#         #print (time.time() - current_time)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_train.append(epoch_loss)
#     with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_train, file)
#     print ("train loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_train, "k^")
#     plt.plot(epoch_losses_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train loss per epoch")
#     plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_train.append(epoch_loss)
#     with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_train, file)
#     print ("train class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_train, "k^")
#     plt.plot(epoch_losses_class_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_train.append(epoch_loss)
#     with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_train, file)
#     print ("train regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_train, "k^")
#     plt.plot(epoch_losses_regr_train, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("train regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
#     plt.close(1)
#
#     print ("####")
#
#     ################################################################################
#     # val:
#     ################################################################################
#     network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
#     batch_losses = []
#     batch_losses_class = []
#     batch_losses_regr = []
#     for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
#         with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
#             imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
#             labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))
#
#             outputs = network(imgs)
#             outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
#             outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))
#             outputs_class = outputs_class.permute(0, 2, 1) # (shape: (batch_size, num_anchors, num_classes))
#
#             labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
#             labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
#             labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))
#
#             # NOTE! NOTE! I use contiguous() here!
#             outputs_regr = outputs_regr.contiguous().view(-1, 4) # (shape: (batch_size*num_anchors, 4))
#             outputs_class = outputs_class.contiguous().view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))
#
#             # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
#             # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
#             # -1 should be ignored for classification)
#
#             ########################################################################
#             # # compute the regression loss:
#             ########################################################################
#             # remove entries which should be ignored (-1 and 0):
#             mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
#             mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
#             labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))
#
#             num_foreground_anchors = float(labels_regr.size()[0])
#
#             if step == 0:
#                 print ("num_foreground_anchors:")
#                 print (num_foreground_anchors)
#
#             loss_regr = regression_loss_func(outputs_regr, labels_regr)
#
#             loss_regr_value = loss_regr.data.cpu().numpy()
#             batch_losses_regr.append(loss_regr_value)
#
#             if step == 0:
#                 print ("outputs_regr.data.cpu().numpy():")
#                 print (outputs_regr.data.cpu().numpy())
#                 print (outputs_regr.data.cpu().numpy().shape)
#                 print ("labels_regr.data.cpu().numpy():")
#                 print (labels_regr.data.cpu().numpy())
#                 print (labels_regr.data.cpu().numpy().shape)
#
#             # ########################################################################
#             # # # compute the classification loss:
#             # ########################################################################
#             # mask = labels_class > -1
#             # mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             # outputs_class = outputs_class[mask, :] # (shape: (num_anchors, num_classes))
#             # labels_class = labels_class[mask] # (shape: (num_anchors, ))
#             #
#             # labels_class_onehot = onehot_embed(labels_class, num_classes) # (shape: (num_class_anchors, num_classes))
#             # CE = -labels_class_onehot*F.log_softmax(outputs_class, dim=1) # (shape: (num_class_anchors, num_classes))
#             # weight = labels_class_onehot*torch.pow(1 - F.softmax(outputs_class, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#             # loss_class = weight*CE # (shape: (num_class_anchors, num_classes))
#             # loss_class, _ = torch.max(loss_class, dim=1) # (shape: (num_class_anchors, ))
#             # loss_class = torch.mean(loss_class)
#             #
#             # loss_class_value = loss_class.data.cpu().numpy()
#             # batch_losses_class.append(loss_class_value)
#             #
#             # if step == 0:
#             #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#             #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#             #     print (outputs_class.data.cpu().numpy().shape)
#             #     print ("labels_class.data.cpu().numpy():")
#             #     print (labels_class.data.cpu().numpy())
#             #     print (labels_class.data.cpu().numpy().shape)
#             #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#             #     print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#             #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#             #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the classification loss:
#             ########################################################################
#             mask_background = labels_class == 0
#             mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
#             labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))
#
#             mask_foreground = labels_class > 0
#             mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
#             outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_foreground_anchors, num_classes))
#             labels_class_foreground = labels_class[mask_foreground] # (shape: (num_foreground_anchors, ))
#
#             num_foreground_anchors = float(labels_class_foreground.size()[0])
#             num_background_anchors = float(labels_class_background.size()[0])
#
#             if step == 0:
#                 print ("num_foreground_anchors:")
#                 print (num_foreground_anchors)
#                 print ("num_background_anchors:")
#                 print (num_background_anchors)
#
#             labels_class_onehot_foreground = onehot_embed(labels_class_foreground, num_classes) # (shape: (num_class_anchors, num_classes))
#             CE_foreground = -labels_class_onehot_foreground*F.log_softmax(outputs_class_foreground, dim=1) # (shape: (num_class_anchors, num_classes))
#             weight_foreground = class_weights_foreground*labels_class_onehot_foreground*torch.pow(1 - F.softmax(outputs_class_foreground, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#             loss_class_foreground = weight_foreground*CE_foreground # (shape: (num_class_anchors, num_classes))
#             loss_class_foreground, _ = torch.max(loss_class_foreground, dim=1) # (shape: (num_class_anchors, ))
#             loss_class_foreground = torch.mean(loss_class_foreground)
#
#             labels_class_onehot_background = onehot_embed(labels_class_background, num_classes) # (shape: (num_class_anchors, num_classes))
#             CE_background = -labels_class_onehot_background*F.log_softmax(outputs_class_background, dim=1) # (shape: (num_class_anchors, num_classes))
#             weight_background = labels_class_onehot_background*torch.pow(1 - F.softmax(outputs_class_background, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
#             loss_class_background = weight_background*CE_background # (shape: (num_class_anchors, num_classes))
#             loss_class_background, _ = torch.max(loss_class_background, dim=1) # (shape: (num_class_anchors, ))
#             loss_class_background = torch.mean(loss_class_background)
#
#             loss_class = loss_class_foreground + lambda_value_neg*loss_class_background
#
#             loss_class_value = loss_class.data.cpu().numpy()
#             batch_losses_class.append(loss_class_value)
#
#             if step == 0:
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
#                 print (outputs_class.data.cpu().numpy().shape)
#                 print ("labels_class.data.cpu().numpy():")
#                 print (labels_class.data.cpu().numpy())
#                 print (labels_class.data.cpu().numpy().shape)
#                 print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#                 print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
#                 print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
#
#             ########################################################################
#             # # compute the total loss:
#             ########################################################################
#             loss = loss_class + lambda_value*loss_regr
#
#             loss_value = loss.data.cpu().numpy()
#             batch_losses.append(loss_value)
#
#     epoch_loss = np.mean(batch_losses)
#     epoch_losses_val.append(epoch_loss)
#     with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_val, file)
#     print ("val loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_val, "k^")
#     plt.plot(epoch_losses_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val loss per epoch")
#     plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_class)
#     epoch_losses_class_val.append(epoch_loss)
#     with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_class_val, file)
#     print ("val class loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_class_val, "k^")
#     plt.plot(epoch_losses_class_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val class loss per epoch")
#     plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
#     plt.close(1)
#
#     epoch_loss = np.mean(batch_losses_regr)
#     epoch_losses_regr_val.append(epoch_loss)
#     with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
#         pickle.dump(epoch_losses_regr_val, file)
#     print ("val regr loss: %g" % epoch_loss)
#     plt.figure(1)
#     plt.plot(epoch_losses_regr_val, "k^")
#     plt.plot(epoch_losses_regr_val, "k")
#     plt.ylabel("loss")
#     plt.xlabel("epoch")
#     plt.title("val regr loss per epoch")
#     plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
#     plt.close(1)
#
#     # save the model weights to disk:
#     checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
#     torch.save(network.state_dict(), checkpoint_path)

from datasets import DatasetMoreAugmentation2, DatasetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from retinanet import RetinaNet

from utils import onehot_embed, init_weights, add_weight_decay

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

import time

# NOTE! change this to not overwrite all log data when you train the model:
model_id = "9_2_2_2_"

num_epochs = 1000
batch_size = 16
learning_rate = 0.0001

lambda_value = 100.0 # (loss weight)
lambda_value_neg = 80.0

gamma = 2.0

network = RetinaNet(model_id, project_dir="/root/retinanet").cuda()

num_classes = network.num_classes

train_dataset = DatasetMoreAugmentation2(kitti_data_path="/root/3DOD_thesis/data/kitti",
                                    kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                                    type="train")
val_dataset = DatasetEval(kitti_data_path="/root/3DOD_thesis/data/kitti",
                          kitti_meta_path="/root/3DOD_thesis/data/kitti/meta",
                          type="val")

num_train_batches = int(len(train_dataset)/batch_size)
num_val_batches = int(len(val_dataset)/batch_size)
print ("num_train_batches:", num_train_batches)
print ("num_val_batches:", num_val_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, shuffle=False,
                                         num_workers=4)

regression_loss_func = nn.SmoothL1Loss()

params = add_weight_decay(network, l2_value=0.0001)
optimizer = torch.optim.Adam(params, lr=learning_rate)

epoch_losses_train = []
epoch_losses_class_train = []
epoch_losses_regr_train = []
epoch_losses_val = []
epoch_losses_class_val = []
epoch_losses_regr_val = []
for epoch in range(num_epochs):
    print ("###########################")
    print ("######## NEW EPOCH ########")
    print ("###########################")
    print ("epoch: %d/%d" % (epoch+1, num_epochs))

    ################################################################################
    # train:
    ################################################################################
    network.train() # (set in training mode, this affects BatchNorm and dropout)
    batch_losses = []
    batch_losses_class = []
    batch_losses_regr = []
    for step, (imgs, labels_regr, labels_class) in enumerate(train_loader):
        #current_time = time.time()

        imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
        labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))

        outputs = network(imgs)
        outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
        outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))
        outputs_class = outputs_class.permute(0, 2, 1) # (shape: (batch_size, num_anchors, num_classes))

        labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
        labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
        labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
        labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))

        # NOTE! NOTE! I use contiguous() here!
        outputs_regr = outputs_regr.contiguous().view(-1, 4) # (shape: (batch_size*num_anchors, 4))
        outputs_class = outputs_class.contiguous().view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))

        # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
        # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
        # -1 should be ignored for classification)

        ########################################################################
        # # compute the regression loss:
        ########################################################################
        # remove entries which should be ignored (-1 and 0):
        mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
        mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
        outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
        labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))

        num_foreground_anchors = float(labels_regr.size()[0])

        if step == 0:
            print ("num_foreground_anchors:")
            print (num_foreground_anchors)

        loss_regr = regression_loss_func(outputs_regr, labels_regr)

        loss_regr_value = loss_regr.data.cpu().numpy()
        batch_losses_regr.append(loss_regr_value)

        if step == 0:
            print ("outputs_regr.data.cpu().numpy():")
            print (outputs_regr.data.cpu().numpy())
            print (outputs_regr.data.cpu().numpy().shape)
            print ("labels_regr.data.cpu().numpy():")
            print (labels_regr.data.cpu().numpy())
            print (labels_regr.data.cpu().numpy().shape)

        # ########################################################################
        # # # compute the classification loss:
        # ########################################################################
        # mask = labels_class > -1
        # mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
        # outputs_class = outputs_class[mask, :] # (shape: (num_anchors, num_classes))
        # labels_class = labels_class[mask] # (shape: (num_anchors, ))
        #
        # labels_class_onehot = onehot_embed(labels_class, num_classes) # (shape: (num_class_anchors, num_classes))
        # CE = -labels_class_onehot*F.log_softmax(outputs_class, dim=1) # (shape: (num_class_anchors, num_classes))
        # weight = labels_class_onehot*torch.pow(1 - F.softmax(outputs_class, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
        # loss_class = weight*CE # (shape: (num_class_anchors, num_classes))
        # loss_class, _ = torch.max(loss_class, dim=1) # (shape: (num_class_anchors, ))
        # loss_class = torch.mean(loss_class)
        #
        # loss_class_value = loss_class.data.cpu().numpy()
        # batch_losses_class.append(loss_class_value)
        #
        # if step == 0:
        #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
        #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
        #     print (outputs_class.data.cpu().numpy().shape)
        #     print ("labels_class.data.cpu().numpy():")
        #     print (labels_class.data.cpu().numpy())
        #     print (labels_class.data.cpu().numpy().shape)
        #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
        #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
        #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
        #     print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
        #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
        #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)

        ########################################################################
        # # compute the classification loss:
        ########################################################################
        mask_background = labels_class == 0
        mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
        outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
        labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))

        mask_foreground = labels_class > 0
        mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
        outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_foreground_anchors, num_classes))
        labels_class_foreground = labels_class[mask_foreground] # (shape: (num_foreground_anchors, ))

        num_foreground_anchors = float(labels_class_foreground.size()[0])
        num_background_anchors = float(labels_class_background.size()[0])

        if step == 0:
            print ("num_foreground_anchors:")
            print (num_foreground_anchors)
            print ("num_background_anchors:")
            print (num_background_anchors)

        labels_class_onehot_foreground = onehot_embed(labels_class_foreground, num_classes) # (shape: (num_class_anchors, num_classes))
        CE_foreground = -labels_class_onehot_foreground*F.log_softmax(outputs_class_foreground, dim=1) # (shape: (num_class_anchors, num_classes))
        weight_foreground = labels_class_onehot_foreground*torch.pow(1 - F.softmax(outputs_class_foreground, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
        loss_class_foreground = weight_foreground*CE_foreground # (shape: (num_class_anchors, num_classes))
        loss_class_foreground, _ = torch.max(loss_class_foreground, dim=1) # (shape: (num_class_anchors, ))
        loss_class_foreground = torch.mean(loss_class_foreground)

        labels_class_onehot_background = onehot_embed(labels_class_background, num_classes) # (shape: (num_class_anchors, num_classes))
        CE_background = -labels_class_onehot_background*F.log_softmax(outputs_class_background, dim=1) # (shape: (num_class_anchors, num_classes))
        weight_background = labels_class_onehot_background*torch.pow(1 - F.softmax(outputs_class_background, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
        loss_class_background = weight_background*CE_background # (shape: (num_class_anchors, num_classes))
        loss_class_background, _ = torch.max(loss_class_background, dim=1) # (shape: (num_class_anchors, ))
        loss_class_background = torch.mean(loss_class_background)

        loss_class = loss_class_foreground + lambda_value_neg*loss_class_background

        loss_class_value = loss_class.data.cpu().numpy()
        batch_losses_class.append(loss_class_value)

        if step == 0:
            print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
            print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
            print (outputs_class.data.cpu().numpy().shape)
            print ("labels_class.data.cpu().numpy():")
            print (labels_class.data.cpu().numpy())
            print (labels_class.data.cpu().numpy().shape)
            print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
            print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
            print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
            print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
            print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
            print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)

        ########################################################################
        # # compute the total loss:
        ########################################################################
        loss = loss_class + lambda_value*loss_regr

        loss_value = loss.data.cpu().numpy()
        batch_losses.append(loss_value)

        ########################################################################
        # optimization step:
        ########################################################################
        optimizer.zero_grad() # (reset gradients)
        loss.backward() # (compute gradients)
        optimizer.step() # (perform optimization step)

        #print (time.time() - current_time)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_train.append(epoch_loss)
    with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_train, file)
    print ("train loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_train, "k^")
    plt.plot(epoch_losses_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train loss per epoch")
    plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_class)
    epoch_losses_class_train.append(epoch_loss)
    with open("%s/epoch_losses_class_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_class_train, file)
    print ("train class loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_class_train, "k^")
    plt.plot(epoch_losses_class_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train class loss per epoch")
    plt.savefig("%s/epoch_losses_class_train.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_regr)
    epoch_losses_regr_train.append(epoch_loss)
    with open("%s/epoch_losses_regr_train.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_regr_train, file)
    print ("train regr loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_regr_train, "k^")
    plt.plot(epoch_losses_regr_train, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("train regr loss per epoch")
    plt.savefig("%s/epoch_losses_regr_train.png" % network.model_dir)
    plt.close(1)

    print ("####")

    ################################################################################
    # val:
    ################################################################################
    network.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
    batch_losses = []
    batch_losses_class = []
    batch_losses_regr = []
    for step, (imgs, labels_regr, labels_class, img_ids) in enumerate(val_loader):
        with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_heigth, img_width))
            labels_regr = Variable(labels_regr).cuda() # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
            labels_class = Variable(labels_class).cuda() # (shape: (batch_size, num_anchors))

            outputs = network(imgs)
            outputs_regr = outputs[0] # (shape: (batch_size, num_anchors, 4)) (x_resid, y_resid, w_resid, h_resid)
            outputs_class = outputs[1] # (shape: (batch_size, num_classes, num_anchors))
            outputs_class = outputs_class.permute(0, 2, 1) # (shape: (batch_size, num_anchors, num_classes))

            labels_regr = labels_regr.view(-1, 4) # (shape: (batch_size*num_anchors, 4))
            labels_class = labels_class.view(-1, 1) # (shape: (batch_size*num_anchors, 1))
            labels_class = labels_class.squeeze() # (shape: (batch_size*num_anchors, ))
            labels_class = Variable(labels_class.data.type(torch.LongTensor)).cuda() # (shape: (batch_size*num_anchors, ))

            # NOTE! NOTE! I use contiguous() here!
            outputs_regr = outputs_regr.contiguous().view(-1, 4) # (shape: (batch_size*num_anchors, 4))
            outputs_class = outputs_class.contiguous().view(-1, num_classes) # (shape: (batch_size*num_anchors, num_classes))

            # (entries in labels_class are in {-1, 0, 1, 2, 3, ..., num_classes-1},
            # where -1: ignore, 0: background. -1 and 0 should be ignored for regression,
            # -1 should be ignored for classification)

            ########################################################################
            # # compute the regression loss:
            ########################################################################
            # remove entries which should be ignored (-1 and 0):
            mask = labels_class > 0 # (shape: (batch_size*num_anchors, ), entries to be ignored are 0, the rest are 1)
            mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
            outputs_regr = outputs_regr[mask, :] # (shape: (num_regr_anchors, 4))
            labels_regr = labels_regr[mask, :] # (shape: (num_regr_anchors, 4))

            num_foreground_anchors = float(labels_regr.size()[0])

            if step == 0:
                print ("num_foreground_anchors:")
                print (num_foreground_anchors)

            loss_regr = regression_loss_func(outputs_regr, labels_regr)

            loss_regr_value = loss_regr.data.cpu().numpy()
            batch_losses_regr.append(loss_regr_value)

            if step == 0:
                print ("outputs_regr.data.cpu().numpy():")
                print (outputs_regr.data.cpu().numpy())
                print (outputs_regr.data.cpu().numpy().shape)
                print ("labels_regr.data.cpu().numpy():")
                print (labels_regr.data.cpu().numpy())
                print (labels_regr.data.cpu().numpy().shape)

            # ########################################################################
            # # # compute the classification loss:
            # ########################################################################
            # mask = labels_class > -1
            # mask = mask.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
            # outputs_class = outputs_class[mask, :] # (shape: (num_anchors, num_classes))
            # labels_class = labels_class[mask] # (shape: (num_anchors, ))
            #
            # labels_class_onehot = onehot_embed(labels_class, num_classes) # (shape: (num_class_anchors, num_classes))
            # CE = -labels_class_onehot*F.log_softmax(outputs_class, dim=1) # (shape: (num_class_anchors, num_classes))
            # weight = labels_class_onehot*torch.pow(1 - F.softmax(outputs_class, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
            # loss_class = weight*CE # (shape: (num_class_anchors, num_classes))
            # loss_class, _ = torch.max(loss_class, dim=1) # (shape: (num_class_anchors, ))
            # loss_class = torch.mean(loss_class)
            #
            # loss_class_value = loss_class.data.cpu().numpy()
            # batch_losses_class.append(loss_class_value)
            #
            # if step == 0:
            #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
            #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
            #     print (outputs_class.data.cpu().numpy().shape)
            #     print ("labels_class.data.cpu().numpy():")
            #     print (labels_class.data.cpu().numpy())
            #     print (labels_class.data.cpu().numpy().shape)
            #     print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
            #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
            #     print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
            #     print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
            #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
            #     print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)

            ########################################################################
            # # compute the classification loss:
            ########################################################################
            mask_background = labels_class == 0
            mask_background = mask_background.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
            outputs_class_background = outputs_class[mask_background, :] # (shape: (num_background_anchors, num_classes))
            labels_class_background = labels_class[mask_background] # (shape: (num_background_anchors, ))

            mask_foreground = labels_class > 0
            mask_foreground = mask_foreground.type(torch.ByteTensor).cuda() # (NOTE! ByteTensor is needed for this to act as a selction mask)
            outputs_class_foreground = outputs_class[mask_foreground, :] # (shape: (num_foreground_anchors, num_classes))
            labels_class_foreground = labels_class[mask_foreground] # (shape: (num_foreground_anchors, ))

            num_foreground_anchors = float(labels_class_foreground.size()[0])
            num_background_anchors = float(labels_class_background.size()[0])

            if step == 0:
                print ("num_foreground_anchors:")
                print (num_foreground_anchors)
                print ("num_background_anchors:")
                print (num_background_anchors)

            labels_class_onehot_foreground = onehot_embed(labels_class_foreground, num_classes) # (shape: (num_class_anchors, num_classes))
            CE_foreground = -labels_class_onehot_foreground*F.log_softmax(outputs_class_foreground, dim=1) # (shape: (num_class_anchors, num_classes))
            weight_foreground = labels_class_onehot_foreground*torch.pow(1 - F.softmax(outputs_class_foreground, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
            loss_class_foreground = weight_foreground*CE_foreground # (shape: (num_class_anchors, num_classes))
            loss_class_foreground, _ = torch.max(loss_class_foreground, dim=1) # (shape: (num_class_anchors, ))
            loss_class_foreground = torch.mean(loss_class_foreground)

            labels_class_onehot_background = onehot_embed(labels_class_background, num_classes) # (shape: (num_class_anchors, num_classes))
            CE_background = -labels_class_onehot_background*F.log_softmax(outputs_class_background, dim=1) # (shape: (num_class_anchors, num_classes))
            weight_background = labels_class_onehot_background*torch.pow(1 - F.softmax(outputs_class_background, dim=1), gamma) # (shape: (num_class_anchors, num_classes))
            loss_class_background = weight_background*CE_background # (shape: (num_class_anchors, num_classes))
            loss_class_background, _ = torch.max(loss_class_background, dim=1) # (shape: (num_class_anchors, ))
            loss_class_background = torch.mean(loss_class_background)

            loss_class = loss_class_foreground + lambda_value_neg*loss_class_background

            loss_class_value = loss_class.data.cpu().numpy()
            batch_losses_class.append(loss_class_value)

            if step == 0:
                print ("F.softmax(outputs_class, dim=1).data.cpu().numpy():")
                print (F.softmax(outputs_class, dim=1).data.cpu().numpy())
                print (outputs_class.data.cpu().numpy().shape)
                print ("labels_class.data.cpu().numpy():")
                print (labels_class.data.cpu().numpy())
                print (labels_class.data.cpu().numpy().shape)
                print ("F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
                print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
                print (F.softmax(outputs_class, dim=1).data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)
                print ("labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0]:")
                print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0])
                print (labels_class.data.cpu().numpy()[labels_class.data.cpu().numpy() > 0].shape)

            ########################################################################
            # # compute the total loss:
            ########################################################################
            loss = loss_class + lambda_value*loss_regr

            loss_value = loss.data.cpu().numpy()
            batch_losses.append(loss_value)

    epoch_loss = np.mean(batch_losses)
    epoch_losses_val.append(epoch_loss)
    with open("%s/epoch_losses_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_val, file)
    print ("val loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_val, "k^")
    plt.plot(epoch_losses_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val loss per epoch")
    plt.savefig("%s/epoch_losses_val.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_class)
    epoch_losses_class_val.append(epoch_loss)
    with open("%s/epoch_losses_class_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_class_val, file)
    print ("val class loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_class_val, "k^")
    plt.plot(epoch_losses_class_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val class loss per epoch")
    plt.savefig("%s/epoch_losses_class_val.png" % network.model_dir)
    plt.close(1)

    epoch_loss = np.mean(batch_losses_regr)
    epoch_losses_regr_val.append(epoch_loss)
    with open("%s/epoch_losses_regr_val.pkl" % network.model_dir, "wb") as file:
        pickle.dump(epoch_losses_regr_val, file)
    print ("val regr loss: %g" % epoch_loss)
    plt.figure(1)
    plt.plot(epoch_losses_regr_val, "k^")
    plt.plot(epoch_losses_regr_val, "k")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title("val regr loss per epoch")
    plt.savefig("%s/epoch_losses_regr_val.png" % network.model_dir)
    plt.close(1)

    # save the model weights to disk:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
