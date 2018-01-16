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
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + lung

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.

    # IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

class CxrDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self):
        super(CxrDataset, self).__init__()

        self.image_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/cxr/images'
        self.mask_dir = '/media/Disk/wangfuyu/Mask_RCNN/data/cxr/masks'

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
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        filename = info['filename']
        count = info['instances']

        mask_path = os.path.join(self.mask_dir, filename + '.png')
        mask = skimage.io.imread(mask_path)

        instance_masks = []
        class_ids = []

        for index in range(1, count + 1):
            m = (mask == index).astype(np.uint8)

            instance_masks.append(m)
            class_ids.append(1)

        instance_masks = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids, dtype=np.int32)

        return instance_masks, class_ids


