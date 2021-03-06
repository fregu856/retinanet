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

9_2_2 epoch 1000 on val_random:
```
car easy detection 0.8073124545454544
car moderate detection 0.7310638181818182
car hard detection 0.6628736363636364
=================
pedestrian easy detection 0.6576356363636363
pedestrian moderate detection 0.5871550909090909
pedestrian hard detection 0.541631
=================
cyclist easy detection 0.5891685454545453
cyclist moderate detection 0.5478943636363638
cyclist hard detection 0.5262490000000001
```

9_2_3 epoch 520 on val_random:
```
car easy detection 0.5915137272727273
car moderate detection 0.513204
car hard detection 0.48550690909090916
=================
pedestrian easy detection 0.458893
pedestrian moderate detection 0.4383350909090908
pedestrian hard detection 0.414931
=================
cyclist easy detection 0.5274312727272727
cyclist moderate detection 0.4361208181818182
cyclist hard detection 0.4332996363636363
```

9_2_3 epoch 1000 on val_random:
```
car easy detection 0.7455983636363637
car moderate detection 0.6282966363636363
car hard detection 0.576304909090909
=================
pedestrian easy detection 0.5058948181818181
pedestrian moderate detection 0.4996721818181818
pedestrian hard detection 0.46471009090909093
=================
cyclist easy detection 0.5217887272727273
cyclist moderate detection 0.4704944545454545
cyclist hard detection 0.45830736363636365
```

9_2_2_2 epoch 380 on val_random:
```
car easy detection 0.8502569090909091
car moderate detection 0.7965869090909091
car hard detection 0.742355
=================
pedestrian easy detection 0.593295
pedestrian moderate detection 0.5470084545454544
pedestrian hard detection 0.49333663636363634
=================
cyclist easy detection 0.5827274545454546
cyclist moderate detection 0.5152459999999999
cyclist hard detection 0.5197565454545455
```

9_2_3_2 epoch 380 on val_random:
```
car easy detection 0.761842090909091
car moderate detection 0.6412636363636364
car hard detection 0.6014667272727272
=================
pedestrian easy detection 0.49154572727272733
pedestrian moderate detection 0.4674940000000001
pedestrian hard detection 0.43318654545454544
=================
cyclist easy detection 0.5239035454545454
cyclist moderate detection 0.4835083636363636
cyclist hard detection 0.4837232727272728
```

9_2_2_2 epoch 390 on test:
```
Car (Detection)	75.21 %	55.68 %	48.40 %
Pedestrian (Detection)	31.69 %	25.36 %	22.16 %
Cyclist (Detection)	25.21 %	17.69 %	16.19 %
```

15 epoch 50 on val (thresh=0.7):
```
car easy detection 0.7796202727272727
car moderate detection 0.6389526363636363
car hard detection 0.5715755454545455
=================
pedestrian easy detection 0.3986153636363637
pedestrian moderate detection 0.3942484545454546
pedestrian hard detection 0.32656490909090913
=================
cyclist easy detection 0.4149840909090909
cyclist moderate detection 0.2773230909090909
cyclist hard detection 0.27941445454545455
```

15 epoch 50 on val (thresh=0.25):
```
car easy detection 0.7796202727272727
car moderate detection 0.6757345454545454
car hard detection 0.6126170000000001
=================
pedestrian easy detection 0.5041476363636364
pedestrian moderate detection 0.4443453636363637
pedestrian hard detection 0.41797245454545456
=================
cyclist easy detection 0.4392936363636364
cyclist moderate detection 0.29614972727272726
cyclist hard detection 0.27511281818181815
```

15 epoch 200 on val (thresh=0.7):
```
car easy detection 0.8002345454545453
car moderate detection 0.6468827272727272
car hard detection 0.579370090909091
=================
pedestrian easy detection 0.481072
pedestrian moderate detection 0.4121746363636363
pedestrian hard detection 0.40411963636363635
=================
cyclist easy detection 0.2900389090909092
cyclist moderate detection 0.18611536363636363
cyclist hard detection 0.20336909090909092
```

16 epoch 150 on val (thresh=0.7):
```
car easy detection 0.6245824545454545
car moderate detection 0.4797283636363636
car hard detection 0.43074445454545457
=================
pedestrian easy detection 0.2845867272727273
pedestrian moderate detection 0.28918472727272726
pedestrian hard detection 0.2277243636363636
=================
cyclist easy detection 0.07169636363636364
cyclist moderate detection 0.11331927272727273
cyclist hard detection 0.11395054545454544
```
