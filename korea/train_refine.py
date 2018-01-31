import os
import model_refine as model
from cxr import *

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

ROOT_DIR = os.getcwd()

# Directory to save weights
MODEL_DIR = os.path.join(ROOT_DIR, "weights")

UNET_MODEL_PATH = os.path.join(MODEL_DIR, "crop_preserve", "256_ch_512_size.hdf5")


config = CxrConfig()


class TrainConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 0.])
    NUM_CLASSES = 1

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.000

    initial_channel = 8
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 512

    def get_weights_name(self):
        weights_name = os.path.join(MODEL_DIR, '', 'crop_preserve',
                                    str(self.initial_channel * 32) + '_ch_'
                                    + str(self.IMAGE_MAX_DIM) + '_size_second') + '.hdf5'
        return weights_name


config = TrainConfig()
config.display()

# Training dataset
dataset_train = CxrDataset()
dataset_train.load_cxr(txt='/media/Disk/wangfuyu/Mask_RCNN/crop_results/train_id.txt')
dataset_train.prepare()

# val dataset
dataset_val= CxrDataset()
dataset_val.load_cxr(txt='/media/Disk/wangfuyu/Mask_RCNN/crop_results/val_id.txt')
dataset_val.prepare()

# Create model in training mode
model = model.UNet(config)


model.load_weights(UNET_MODEL_PATH)
#
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=10,
#             layers="heads")
#
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE / 10,
#             epochs=30,
#             layers="all")

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 100,
            epochs=20,
            layers="all")
