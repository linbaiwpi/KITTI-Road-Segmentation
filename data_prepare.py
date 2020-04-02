import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

base_dir = './data_road_ext/validation/'
valid_gtdir = base_dir + 'semantic/'
valid_gtlist = os.listdir(valid_gtdir)

'''
gt_path = './data_road_ext/training/gt_image_2/um_road_000000.png'
gt_np = np.array(Image.open(gt_path))
print(gt_np.shape)
plt.imshow(gt_np)
plt.show()
print(gt_np[0,0,:])
print(gt_np[300,600,:])

sem_path = './data_road_ext/validation/semantic/000000_10.png'
sem_np = np.array(Image.open(sem_path))
print(sem_np.shape)
plt.imshow(sem_np)
plt.show()
print(sem_np[0,0])
print(sem_np[300,600])
'''

for gt_name in valid_gtlist:
    gt_s = Image.open(valid_gtdir + gt_name)
    gt_s = np.array(gt_s)
    print(gt_s.shape)
    IMG_H = gt_s.shape[0]
    IMG_W = gt_s.shape[1]
    gt_ = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for p in range(IMG_H):
        for k in range(IMG_W):
            if gt_s[p, k] == 7:
                gt_[p, k, :] = [255,0,255]
            else:
                gt_[p, k, :] = [255,0,0]
    #gt_[:, :, 1] = 255 - gt_[:, :, 0]
    Image.fromarray(gt_).save(base_dir + 'gt_image_2/' + gt_name)

