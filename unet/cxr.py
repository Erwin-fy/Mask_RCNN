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

import skimage.color
import skimage.io

from config import Config
import utils

class CxrConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cxr"

    # Train on 2 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + lung

    IMAGES_PER_GPU = 1


class CxrDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self):
        super(CxrDataset, self).__init__()

        # self.image_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/cxr/images'
        # self.mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/cxr/binary_masks'
        self.image_dir = '/media/Disk/wangfuyu/Mask_RCNN/crop_results/512_320/images'
        self.mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/crop_results/512_320/masks'
        # self.mrcnn_mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/crop_results/512_320/mrcnn_masks'
        self.mrcnn_mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/unet/results/crop_preserve/512_320/512_320_crop'

    def load_cxr(self, txt):
        """Load Dataset
       txt dataset ids
        """
        # Add classes
        self.add_class("cxr", 1, "lung")

        # Add images
        # read image id from txt

        with open(txt, 'r') as f:
            lines = f.readlines()
            for index, line in enumerate(lines):
                self.add_image(
                    "cxr", image_id=index,
                    path=None,
                    filename=line[0:-1],
                    instances=2)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        filename = self.image_info[image_id]['filename']
        img_path = os.path.join(self.image_dir, filename + '.jpg')

        image = skimage.io.imread(img_path)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        return image

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "cxr":
            return info["instances"]
        else:
            super(CxrDataset, self).image_reference(self, image_id)

    def load_mask(self, image_id):
        filename = self.image_info[image_id]['filename']
        mask_path = os.path.join(self.mask_dir, filename + '.png')

        mask = skimage.io.imread(mask_path)

        mask = np.expand_dims(mask, axis=2)

        return mask

    def load_mrcnn_mask(self, image_id):
        filename = self.image_info[image_id]['filename']
        mrcnn_mask_path = os.path.join(self.mrcnn_mask_dir, filename + '.png')
        mrcnn_mask = skimage.io.imread(mrcnn_mask_path)

        mrcnn_mask = np.expand_dims(mrcnn_mask, axis=2)

        return mrcnn_mask

