import cv2
import math
import os
import re
import keras
from glob import glob

from tensorflow.keras.utils import Sequence

import numpy as np
import scipy.misc
from keras.preprocessing.image import load_img
from imgaug import augmenters as iaa
from PIL import Image

import matplotlib.pyplot as plt


class DataSequence(Sequence):

    def __init__(self, data_dir, batch_size, image_shape, training=True):
        """
        Keras Sequence object to train a model on larger-than-memory data.
            @:param: data_dir: directory in which we have got the kitti images and the corresponding masks
            @:param: batch_size: define the number of training samples to be propagated.
            @:param: image_shape: shape of the input image
        """

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.training = training
        self.image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))
        #print(self.image_paths)       
        print("*****************[DATA INFO]*****************")
        if (training):
            print("Found " + str(len(self.image_paths)) + " training images")
        else:
            print("Found " + str(len(self.image_paths)) + " validation images")
        print("*********************************************")
        
        if (training):
            self.label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
                for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))}
            #glob(os.path.join(data_dir, 'gt_image_2', '*.png'))
        else:
            self.label_paths = {os.path.basename(path): path
                for path in glob(os.path.join(data_dir, 'gt_image_2', '*.png'))}
                    
        #print(self.label_paths)        
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images

                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               ]),
                               iaa.Add((-10, 10), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                               iaa.MultiplyHueAndSaturation((0.5, 1.5)),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return int(math.ceil(len(self.image_paths) / float(self.batch_size)))

    def get_batch_images(self, idx, path_list):
        """

        :param idx: position of the image in the Sequence.
        :param path_list: list that consists of all the image paths
        :return: Retrieve the images in batches
        """
        # Fetch a batch of images from a list of paths
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            # load the image and resize
            # image = load_img(im)
            image = Image.open(im).convert('RGB')
            image = image.resize((self.image_shape[0], self.image_shape[1]), Image.BILINEAR)
            image = np.array(image)
            # image = scipy.misc.imresize(image, (self.image_shape[1], self.image_shape[0]))
            # image = np.array(Image.fromarray(image).resize((self.image_shape[1], self.image_shape[0]),Image.NEAREST))
            # augment the image
            if (self.training):
                image = self.aug_pipe.augment_image(image)

            # print(np.array([image]).shape)
            # Image.fromarray(np.squeeze(np.array([image]))).save('./'+str(im.split('/')[-1]))
            # np.save('image2',np.array([image])/255)
            return np.array([image])/255


    def get_batch_labels(self, idx, path_list):

        """
        Retrieve the masks in batches
        :param idx: position of the mask in the Sequence.
        :param path_list: list that consists of all the mask paths
        :return: mask labels
        """
        # iterate and map the mask labels for the respective images
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            # print(os.path.basename(im))
            # print("=======lalala=======")
            gt_image_file = self.label_paths[os.path.basename(im)]
            # gt_image = load_img(gt_image_file)
            gt_image = Image.open(gt_image_file).convert('RGB')
            gt_image = gt_image.resize((self.image_shape[0], self.image_shape[1]), Image.NEAREST)
            gt_image = np.array(gt_image)
            # gt_image = scipy.misc.imresize(gt_image, (self.image_shape[1], self.image_shape[0]))
            # gt_image = np.array(Image.fromarray(gt_image).resize((self.image_shape[1], self.image_shape[0]),Image.NEAREST))
            background_color = np.array([255, 0, 0])
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
            # print("=====================" + str(np.array([gt_image]).shape))
            # np.save('label2', np.array([gt_image]))
            return np.array([gt_image])

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """
        batch_x = self.get_batch_images(idx, self.image_paths)
        batch_y = self.get_batch_labels(idx, self.image_paths)
        '''
        print("===============check batch==================")
        print(np.squeeze(batch_y).shape)
        #plt.imshow(np.squeeze(batch_x))
        plt.imshow(np.squeeze(batch_y[:,:,:,0]))
        plt.imshow(np.squeeze(batch_y[:,:,:,1]))
        plt.show()
        '''
        return batch_x, batch_y
    
