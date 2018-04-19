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
from cxr import *
import utils
import model as model

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
ROOT_DIR = os.getcwd()


# Directory to save weights
# MODEL_DIR = os.path.join(JSRT_DIR, "logs", "odd")
MODEL_DIR = os.path.join(ROOT_DIR, "logs", "CXR")

NET_MODEL_PATH = os.path.join(MODEL_DIR, "korea_stage1.h5")

RESULTS_PATH = os.path.join(ROOT_DIR, "results", "CXR", "stage1")
isExists=os.path.exists(RESULTS_PATH)
if not isExists:
    os.makedirs(RESULTS_PATH)


config = CxrConfig()


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    concat = False
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    NUM_CLASSES = 2


config = InferenceConfig()
# config.display()


# val dataset
dataset = CxrDataset()
dataset.load_cxr(txt='/media/Disk/wangfuyu/Mask_RCNN/data/cxr/val_id.txt')
dataset.prepare()

net = model.Net(mode='test', config=config, model_dir=MODEL_DIR)
net.load_weights(filepath=NET_MODEL_PATH, by_name=False)
iu = net.evaluation(dataset, RESULTS_PATH)
print(iu.mean())
