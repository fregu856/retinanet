# retinanet

### Train model on *KITTI train*:

- SSH into the paperspace server.
- $ sudo sh start_docker_image.sh
- $ cd --
- $ python retinanet/train_retinanet.py


****
****

Result of running preprocess_data.py:
```
83355580
83267981
75570
7578
4451
[1.42333128231621, 48.33006556954434, 50.2720914823404, 50.36520653297994]
```

After modification:
```
95494
83152
7712
4630
[1.0, 1.5699156534303222, 10.416672919112907, 15.096277253799496]
```
****
****
****

nms_thresh = 0.5
class_thresh = 0.5

9_2___ epoch 520 on val_random:
```
car easy detection 0.7595934545454545
car moderate detection 0.6821118181818183
car hard detection 0.6274337272727273
=================
pedestrian easy detection 0.5281956363636362
pedestrian moderate detection 0.5099803636363637
pedestrian hard detection 0.47057581818181815
=================
cyclist easy detection 0.5775938181818182
cyclist moderate detection 0.506604909090909
cyclist hard detection 0.5054033636363635
```

9_2_2 epoch 140 on val_random:
```
car easy detection 0.6897030909090908
car moderate detection 0.5719970909090908
car hard detection 0.5145053636363636
=================
pedestrian easy detection 0.4633137272727272
pedestrian moderate detection 0.4278780909090909
pedestrian hard detection 0.4030180909090909
=================
cyclist easy detection 0.4406213636363636
cyclist moderate detection 0.41050045454545464
cyclist hard detection 0.39612745454545456
```

9_2_2 epoch 520 on val_random:
```
car easy detection 0.8168949090909091
car moderate detection 0.7367157272727273
car hard detection 0.6661809090909091
=================
pedestrian easy detection 0.6076561818181817
pedestrian moderate detection 0.5453122727272727
pedestrian hard detection 0.5014571818181819
=================
cyclist easy detection 0.558449090909091
cyclist moderate detection 0.5075719090909092
cyclist hard detection 0.48960345454545456
```
