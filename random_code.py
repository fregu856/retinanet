import random
random.shuffle(img_ids)
random.shuffle(img_ids)
random.shuffle(img_ids)
random.shuffle(img_ids)

no_of_imgs = len(img_ids)
train_img_ids = img_ids[:int(no_of_imgs*0.9)]
val_img_ids = img_ids[-int(no_of_imgs*0.1):]

print (no_of_imgs)
print ("train:", len(train_img_ids))
print ("val:", len(val_img_ids))

with open("/root/retinanet/data/synscapes_meta/train_img_ids.pkl", "wb") as file:
    pickle.dump(train_img_ids, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
with open("/root/retinanet/data/synscapes_meta/val_img_ids.pkl", "wb") as file:
    pickle.dump(val_img_ids, file, protocol=2) # (protocol=2 is needed to be able to open this file with python2)
