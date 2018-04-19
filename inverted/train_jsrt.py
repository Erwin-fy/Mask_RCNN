import os

import model as modellib

from jsrt import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
RESNET18_MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_cxr_res18_renet_C5_wave.h5")

config = JsrtConfig()
config.display()

# Training dataset
dataset_train = JsrtDataset()
dataset_train.load_jsrt(txt='/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/odd_id_aug.txt')
dataset_train.prepare()

# val dataset
dataset_val= JsrtDataset()
dataset_val.load_jsrt(txt='/media/Disk/wangfuyu/Mask_RCNN/data/jsrt/even_id.txt')
dataset_val.prepare()

# Create model in training mode
architecture = 'inverted'
net = modellib.Net(mode="training", architecture=architecture,
                   config=config, model_dir=MODEL_DIR)

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
net.train(dataset_train, dataset_val,
          learning_rate=config.LEARNING_RATE,
          epochs=120,
          layers="all")

model_path = os.path.join(MODEL_DIR, architecture+"_256.h5")
net.model.save_weights(model_path)


