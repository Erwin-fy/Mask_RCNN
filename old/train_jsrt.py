import os

import model_res18_wave as modellib

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
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "None"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

elif init_with == "Res18":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(RESNET18_MODEL_PATH, by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=20,
#             layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=120,
            layers="all")

model_path = os.path.join(MODEL_DIR, "res18_renet_C5_GRU_JSRT_Kmeans4_aug.h5")
model.keras_model.save_weights(model_path)


