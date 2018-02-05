import os
import model as model
from cxr import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

ROOT_DIR = os.getcwd()

# Directory to save weights
MODEL_DIR = os.path.join(ROOT_DIR, "logs", "CXR", "256")

# LAST_MODEL_Dir = os.path.join(ROOT_DIR, "logs", "JSRT", "odd", "korea_stage1.h5")

config = CxrConfig()


class TrainConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4
    MEAN_PIXEL = np.array([191.6, 191.6, 191.6])
    NUM_CLASSES = 1

    LEARNING_RATE = 1e-5
    LEARNING_MOMENTUM = 0.
    WEIGHT_DECAY = 0.


config = TrainConfig()
config.display()

# Training dataset
dataset_train = CxrDataset()
dataset_train.load_cxr(txt='/media/Disk/wangfuyu/Mask_RCNN/data/cxr/train_id.txt')
dataset_train.prepare()

# val dataset
dataset_val= CxrDataset()
dataset_val.load_cxr(txt='/media/Disk/wangfuyu/Mask_RCNN/data/cxr/val_id.txt')
dataset_val.prepare()

# Create model in training mode
net = model.Net(mode='train', config=config, model_dir=MODEL_DIR)

# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=10,
#             layers="heads")
#
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=15,
#             layers="all")

# net.load_weights(LAST_MODEL_Dir, by_name=True)
net.train(dataset_train, dataset_val,
          learning_rate=config.LEARNING_RATE,
          epochs=70,
          layers="all")

model_path = os.path.join(MODEL_DIR, "inverted_stage1.h5")
net.model.save_weights(model_path)
