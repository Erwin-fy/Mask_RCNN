"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import numpy as np
import re

import skimage.color
import skimage.io
import cv2

from config import Config
import utils


class JsrtConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "jsrt"

    # Train on 2 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + lung

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.

    # IMAGE_MIN_DIM = 200
    # IMAGE_MAX_DIM = 256
    MEAN_PIXEL = np.array([191.7, 191.7, 191.7])

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    RPN_ANCHOR_RATIOS = [2.2, 0.47, 0.68, 1.95]

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5


class JsrtDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self):
        super(JsrtDataset, self).__init__()

        self.image_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/1024/images'
        self.mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/1024/binary_masks'

    def load_jsrt(self, txt):
        """Load Dataset
       txt dataset ids
        """
        # Add classes
        self.add_class("jsrt", 1, "lung")
        self.add_class("jsrt", 2, "clavicle")
        self.add_class("jsrt", 3, "heart")

        # Add images
        # read image id from txt

        with open(txt, 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                self.add_image(
                    "jsrt", image_id=index,
                    path=None,
                    filename=line[0:-1],
                    instances=5)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        filename = self.image_info[image_id]['filename']
        img_path = os.path.join(self.image_dir, filename + '.jpg')
        image = skimage.io.imread(img_path)

        # Waveform
        length = 128
        image_gray = skimage.color.rgb2gray(image)
        rowWave = np.mean(image_gray, axis=1)
        rowWave = np.expand_dims(rowWave, axis=1)
        rowWave = cv2.resize(rowWave, (1, length))
        rowWave = np.squeeze(rowWave, axis=1)

        colWave = np.mean(image_gray, axis=0)
        colWave = np.expand_dims(colWave, axis=0)
        colWave = cv2.resize(colWave, (length, 1))
        colWave = np.squeeze(colWave, axis=0)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        return image, rowWave, colWave

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "jsrt":
            return info["instances"]
        else:
            super(JsrtDataset, self).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        filename = info['filename']
        count = info['instances']

        heart = skimage.io.imread(os.path.join(self.mask_dir, 'heart', filename + '.png'))
        left_lung = skimage.io.imread(os.path.join(self.mask_dir, 'left_lung', filename + '.png'))
        right_lung = skimage.io.imread(os.path.join(self.mask_dir, 'right_lung', filename + '.png'))
        left_clavicle = skimage.io.imread(os.path.join(self.mask_dir, 'left_clavicle', filename + '.png'))
        right_clavicle = skimage.io.imread(os.path.join(self.mask_dir, 'right_clavicle', filename + '.png'))

        instance_masks = []
        instance_masks.append(left_lung)
        instance_masks.append(right_lung)
        instance_masks.append(left_clavicle)
        instance_masks.append(right_clavicle)
        instance_masks.append(heart)
        instance_masks = np.stack(instance_masks, axis=2)
        class_ids = np.array([1, 1, 2, 2, 3], dtype=np.int32)

        return instance_masks, class_ids

###  JSRT
# class CxrDataset(utils.Dataset):
#     """Generates the shapes synthetic dataset. The dataset consists of simple
#     shapes (triangles, squares, circles) placed randomly on a blank surface.
#     The images are generated on the fly. No file access required.
#     """
#     def __init__(self):
#         super(CxrDataset, self).__init__()
#
#         self.root_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/cxr/256x256/JM'
#         # self.mrcnn_mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/crop_results/512_320/mrcnn_masks'
#
#     def load_cxr(self, txt):
#         """Load Dataset
#        txt dataset ids
#         """
#         # Add classes
#         self.add_class("cxr", 1, "lung")
#
#         # Add images
#         # read image id from txt
#
#         with open(txt, 'r') as f:
#             lines = f.readlines()
#             for index, line in enumerate(lines):
#                 tmp = re.split(' ', line)
#                 self.add_image(
#                     "cxr", image_id=index,
#                     path=None,
#                     img_path=tmp[0],
#                     mask_path=tmp[1][0:-1],
#                     filename=tmp[0],
#                     instances=2)
#
#     def load_image(self, image_id):
#         """Load the specified image and return a [H,W,3] Numpy array.
#         """
#         # Load image
#         # filename = self.image_info[image_id]['filename']
#         # img_path = os.path.join(self.image_dir, filename + '.jpg')
#
#         img_path = os.path.join(self.root_dir, self.image_info[image_id]['img_path'])
#         image = skimage.io.imread(img_path)
#
#         # Waveform
#         length = 128
#         image_gray = skimage.color.rgb2gray(image)
#         rowWave = np.mean(image_gray, axis=1)
#         rowWave = np.expand_dims(rowWave, axis=1)
#         rowWave = cv2.resize(rowWave, (1, length))
#         rowWave = np.squeeze(rowWave, axis=1)
#
#         colWave = np.mean(image_gray, axis=0)
#         colWave = np.expand_dims(colWave, axis=0)
#         colWave = cv2.resize(colWave, (length, 1))
#         colWave = np.squeeze(colWave, axis=0)
#
#         # If grayscale. Convert to RGB for consistency.
#         if image.ndim != 3:
#             image = skimage.color.gray2rgb(image)
#
#         return image, rowWave, colWave
#
#     def image_reference(self, image_id):
#         info = self.image_info[image_id]
#         if info["source"] == "cxr":
#             return info["instances"]
#         else:
#             super(CxrDataset, self).image_reference(self, image_id)
#
#     def load_mask(self, image_id):
#             """Generate instance masks for shapes of the given image ID.
#             """
#             info = self.image_info[image_id]
#             count = info['instances']
#
#             mask_path = os.path.join(self.root_dir, self.image_info[image_id]['mask_path'])
#             mask = skimage.io.imread(mask_path)
#
#             instance_masks = []
#             class_ids = []
#
#             for index in range(1, count + 1):
#                 m = (mask == index).astype(np.uint8)
#
#                 instance_masks.append(m)
#                 class_ids.append(1)
#
#             instance_masks = np.stack(instance_masks, axis=2)
#             class_ids = np.array(class_ids, dtype=np.int32)
#
#             return instance_masks, class_ids
#
#     def load_mrcnn_mask(self, image_id):
#         filename = self.image_info[image_id]['filename']
#         mrcnn_mask_path = os.path.join(self.mrcnn_mask_dir, filename + '.png')
#         mrcnn_mask = skimage.io.imread(mrcnn_mask_path)
#
#         mrcnn_mask = np.expand_dims(mrcnn_mask, axis=2)
#
#         return mrcnn_mask
#
