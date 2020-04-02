#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json

import numpy as np
from PIL import Image
from os import listdir, remove, mkdir
from shutil import rmtree
from os.path import join
import h5py

import tensorflow as tf

from tensorflow.keras.models import Sequential,load_model, Model

import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
import math

import transform2BEV
import matplotlib.pyplot as plt
from src.frontend import Segment

base_dir = './data_road_ext/'
session = base_dir+'training/'
train_img_dir = session+'image_2/'
session = base_dir+'validation/'
valid_img_dir = session+'image_2/'
session = base_dir+'testing/'
test_img_dir = session+'image_2/'

train_vis_dir = 'train_visual/'
valid_vis_dir = 'valid_visual/'
test_vis_dir = 'test_visual/'

img_dir = session+'image_2/'
vis_dir = session+'res_vis/'
res_dir = session+'res_cam/'
BEV_dir = session+'res_BEV/'
cal_dir = session+'calib/'

model_name = './result/UBNet_shorter_upsamplemodel-008-0.916.h5'

train_img_list = [f for f in listdir(train_img_dir) if f.endswith('.png')]
list_length = len(train_img_list)
print(list_length)

try:
    rmtree(train_vis_dir)
except OSError:
    pass
try:
    rmtree(valid_vis_dir)
except OSError:
    pass
try:
    rmtree(test_vis_dir)
except OSError:
    pass

mkdir(train_vis_dir)
mkdir(valid_vis_dir)
mkdir(test_vis_dir)

config_path = './config.json'
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())
# parse the json to retrieve the training configuration
backend = config["model"]["backend"]
input_size = (config["model"]["im_width"], config["model"]["im_height"])
classes = config["model"]["classes"]
train_data_dir = config["train"]["data_directory"]
valid_data_dir = config["valid"]["data_directory"]

# define the model and train
segment = Segment(backend, input_size, classes)

threshold = 0.5

img_vis_dir_list = ['train_visual/', 'valid_visual/', 'test_visual/']
for img_vis_dir in img_vis_dir_list:
    if 'train' in img_vis_dir:
        img_dir = train_img_dir
    elif 'valid' in img_vis_dir:
        img_dir = valid_img_dir
    elif 'test' in img_vis_dir:
        img_dir = test_img_dir
    else:
        img_dir = ''

    img_name_list = [f for f in listdir(img_dir) if f.endswith('.png')]
    for img_name in img_name_list:
        image = np.array(Image.open(img_dir + img_name).resize((input_size[0], input_size[1]), 0))
        x_dset = np.expand_dims(image/255, 0)
        segment.feature_extractor.load_weights(model_name)
        y_dset = segment.feature_extractor.predict(x_dset, batch_size=1, verbose=1)
        y_image = np.zeros((input_size[1], input_size[0]), dtype=np.uint8)
        y_image[:, :] = (y_dset[0, :, :, 0] > threshold) * 255

        print("image[:, :, 1]: "+str(image[:, :, 1].shape))
        print("y_image: "+str(y_image.shape))
        image[:, :, 1] = np.bitwise_or(image[:, :, 0], y_image)
        Image.fromarray(image).save(img_vis_dir+img_name)


'''
image_name = image_list[0]
image = np.array(Image.open(img_dir + image_name).resize((input_size[0], input_size[1]), 0))
# image = np.array(Image.open('um_000001.png'))
# image = np.squeeze(np.load('image2.npy'))
# label = np.squeeze(np.load('label2.npy'))
print(image.shape)
plt.imshow(image)
plt.show()
image = image/255
x_dset = np.expand_dims(image, 0)
segment.feature_extractor.load_weights(model_name)
y_dset = segment.feature_extractor.predict(x_dset, batch_size=1, verbose=1)
print("y_dest.shape: "+str(y_dset.shape))
y_image = np.zeros((input_size[1], input_size[0]), dtype=np.uint8)
y_image[:, :] = (y_dset[0, :, :, 1] > 0.7)*255
y_dset_squ = np.squeeze(y_dset[0, :, :, 0]*255)
print("y_dest.shape: "+str(y_dset_squ.shape))
plt.imshow(y_dset_squ, 'gray')
plt.show()
'''
