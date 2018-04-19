"""
Mask R-CNN
The main Mask R-CNN model implemenetation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import logging
import numpy as np
import tensorflow as tf
import keras

import config
from jsrt import *
import utils
import model_refine as model

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save weights
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# UNET_MODEL_PATH = os.path.join(MODEL_DIR, "Hospital_fm_512ch_input512.hdf5")

RESULTS_PATH = os.path.join(ROOT_DIR, "results", "second", "crop_preserve", "JSRT", "renet_C5_wave", "512_320", "512_320_crop")
# BOX_PATH = os.path.join(ROOT_DIR, "results", "crop_preserve", "JSRT/800", "renet_C5_wave", "512_320", "box")
BOX_PATH = '/media/Disk/wangfuyu/Mask_RCNN/refine/HED/jsrt/renet_C5_GRU_Kmeans4/512_320/box'

isExists=os.path.exists(RESULTS_PATH)
if not isExists:
    os.makedirs(RESULTS_PATH)


config = JsrtConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    MEAN_PIXEL = np.array([191.6, 191.6, 191.6, 0.])
    NUM_CLASSES = 1

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization

    initial_channel = 8
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 512

    def get_weights_name(self):
        weights_name = os.path.join(MODEL_DIR, 'crop_preserve',
                                    str(self.initial_channel * 32) + '_ch_'
                                    + str(self.IMAGE_MAX_DIM) + '_size') + '.hdf5'
        return weights_name

config = InferenceConfig()
# config.display()

# val dataset
dataset = JsrtDataset()
dataset.load_jsrt(txt='/media/Disk/wangfuyu/Mask_RCNN/crop_results/jsrt/renet_C5_GRU_Kmeans4/512_320/val_id.txt')
dataset.prepare()

unet = model.UNet(mode="testing", config=config,
                  model_dir=MODEL_DIR)

# weights_path = os.path.join(MODEL_DIR, "JSRT/crop_preserve/256_ch_512_size", "JSRT.h5")
# unet.load_weights(filepath=weights_path, by_name=False)
# iu = unet.evaluation(dataset, RESULTS_PATH)
# print(iu.mean())

unet.save_box(dataset, BOX_PATH)
# unet.save_results(dataset, RESULTS_PATH)
