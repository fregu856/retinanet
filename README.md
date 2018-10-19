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

9_2___ on val_random:
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
