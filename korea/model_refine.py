import os
import sys
import random
import re
import logging
import numpy as np
import cv2
import scipy.misc
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.regularizers as KR
import keras.engine as KE
import keras.models as KM
from keras import losses


import config
from cxr import *
import utils

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)


############################################################
#  Data Formatting
############################################################

def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

'''
def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
'''


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask = dataset.load_mask(image_id)
    result = dataset.load_mrcnn_mask(image_id)

    # image, window, scale, padding = utils.resize_image(
    #     image,
    #     min_dim=config.IMAGE_MIN_DIM,
    #     max_dim=config.IMAGE_MAX_DIM,
    #     padding=config.IMAGE_PADDING)
    # mask = utils.resize_mask(mask, scale, padding)
    # result = utils.resize_mask(result, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            result = np.fliplr(result)

    image = np.concatenate((image, result), axis=2)

    return image, mask


def data_generator(dataset, config, shuffle=True, augment=True, batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - gt_masks: [batch, height, width].

    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_masks = np.zeros(
                    (batch_size, image.shape[0], image.shape[1], 1))


            # Add to batch
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_masks[b] = gt_masks

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_masks]

                yield inputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


class UNet():
    def __init__(self, config):
        self.config = config
        self.WEIGHT_DECAY = config.WEIGHT_DECAY
        self.input_shape = (config.IMAGE_MAX_DIM, config.IMAGE_MIN_DIM, 4)
        self.model = self.build()

    def log_layers(self):
        layers = self.model.layers
        for layer in layers:
            print(layer.name)

    def build(self, mode='train'):
        inputs = KL.Input(shape=self.input_shape)

        conv1 = KL.Conv2D(8, (3, 3), padding='same', name='input_concat4',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(inputs)
        conv1 = KL.BatchNormalization()(conv1)
        conv1 = KL.Activation('relu')(conv1)
        conv1 = KL.Conv2D(8, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv1)
        conv1 = KL.BatchNormalization()(conv1)
        conv1 = KL.Activation('relu')(conv1)
        pool1 = KL.MaxPool2D((2,2), strides=(2,2))(conv1)
        print(conv1.get_shape(), K.int_shape(conv1)[1:])

        conv2 = KL.Conv2D(16, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool1)
        conv2 = KL.BatchNormalization()(conv2)
        conv2 = KL.Activation('relu')(conv2)
        conv2= KL.Conv2D(16, (3, 3), padding='same',
                         kernel_initializer='glorot_normal',
                         kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv2)
        conv2 = KL.BatchNormalization()(conv2)
        conv2 = KL.Activation('relu')(conv2)
        pool2 = KL.MaxPool2D((2, 2), strides=(2, 2))(conv2)

        conv3 = KL.Conv2D(32, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool2)
        conv3 = KL.BatchNormalization()(conv3)
        conv3 = KL.Activation('relu')(conv3)
        conv3 = KL.Conv2D(32, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv3)
        conv3 = KL.BatchNormalization()(conv3)
        conv3 = KL.Activation('relu')(conv3)
        pool3 = KL.MaxPool2D((2, 2), strides=(2, 2))(conv3)

        conv4 = KL.Conv2D(64, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool3)
        conv4 = KL.BatchNormalization()(conv4)
        conv4 = KL.Activation('relu')(conv4)
        conv4 = KL.Conv2D(64, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv4)
        conv4 = KL.BatchNormalization()(conv4)
        conv4 = KL.Activation('relu')(conv4)
        pool4 = KL.MaxPool2D((2, 2), strides=(2, 2))(conv4)

        conv5 = KL.Conv2D(128, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool4)
        conv5 = KL.BatchNormalization()(conv5)
        conv5 = KL.Activation('relu')(conv5)
        conv5 = KL.Conv2D(128, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv5)
        conv5 = KL.BatchNormalization()(conv5)
        conv5 = KL.Activation('relu')(conv5)
        pool5= KL.MaxPool2D((2, 2), strides=(2, 2))(conv5)

        conv6 = KL.Conv2D(256, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool5)
        conv6 = KL.BatchNormalization()(conv6)
        conv6 = KL.Activation('relu')(conv6)
        conv6 = KL.Conv2D(256, (3, 3), padding='same',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv6)
        conv6 = KL.BatchNormalization()(conv6)
        conv6 = KL.Activation('relu')(conv6)
        # pool6 = KL.MaxPool2D((2, 2), strides=(2, 2))(conv6)
        #
        # conv7 = KL.Conv2D(512, (3, 3), padding='same',
        #                   kernel_initializer='glorot_normal',
        #                   kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool6)
        # conv7 = KL.BatchNormalization()(conv7)
        # conv7 = KL.Activation('relu')(conv7)
        # conv7 = KL.Conv2D(512, (3, 3), padding='same',
        #                   kernel_initializer='glorot_normal',
        #                   kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv7)
        # conv7 = KL.BatchNormalization()(conv7)
        # conv7 = KL.Activation('relu')(conv7)
        # pool7 = KL.MaxPool2D((2, 2), strides=(2, 2))(conv7)
        #
        # conv8 = KL.Conv2D(1024, (3, 3), padding='same',
        #                   kernel_initializer='glorot_normal',
        #                   kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(pool7)
        # conv8 = KL.BatchNormalization()(conv8)
        # conv8 = KL.Activation('relu')(conv8)
        # conv8 = KL.Conv2D(1024, (3, 3), padding='same',
        #                   kernel_initializer='glorot_normal',
        #                   kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv8)
        # conv8 = KL.BatchNormalization()(conv8)
        # conv8 = KL.Activation('relu')(conv8)
        #
        # deconv1 = KL.Conv2DTranspose(512, (2, 2), strides=(2,2), padding='valid',
        #                              kernel_initializer='glorot_normal',
        #                              kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv8)
        # deconv1 = KL.Activation('relu')(deconv1)
        # deconv1 = KL.Concatenate(axis=3)([conv7, deconv1])
        # deconv1 = KL.Conv2D(512, (3, 3), padding='same',
        #                   kernel_initializer='glorot_normal',
        #                   kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv1)
        # deconv1 = KL.BatchNormalization()(deconv1)
        # deconv1 = KL.Activation('relu')(deconv1)
        # deconv1 = KL.Conv2D(512, (3, 3), padding='same',
        #                   kernel_initializer='glorot_normal',
        #                   kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv1)
        # deconv1 = KL.BatchNormalization()(deconv1)
        # deconv1 = KL.Activation('relu')(deconv1)

        # deconv2 = KL.Conv2DTranspose(256, (2, 2), strides=(2,2), padding='valid',
        #                              kernel_initializer='glorot_normal',
        #                              kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv7)
        # deconv2 = KL.Activation('relu')(deconv2)
        # deconv2 = KL.Concatenate(axis=3)([conv6, deconv2])
        # deconv2 = KL.Conv2D(256, (3, 3), padding='same',
        #                    kernel_initializer='glorot_normal',
        #                    kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv2)
        # deconv2 = KL.BatchNormalization()(deconv2)
        # deconv2 = KL.Activation('relu')(deconv2)
        # deconv2 = KL.Conv2D(256, (3, 3), padding='same',
        #                    kernel_initializer='glorot_normal',
        #                    kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv2)
        # deconv2 = KL.BatchNormalization()(deconv2)
        # deconv2 = KL.Activation('relu')(deconv2)

        # deconv3 = KL.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid',
        #                               kernel_initializer='glorot_normal',
        #                               kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv6)
        # deconv3 = KL.Activation('relu')(deconv3)
        # deconv3 = KL.Concatenate(axis=3)([conv5, deconv3])
        # deconv3 = KL.Conv2D(128, (3, 3), padding='same',
        #                    kernel_initializer='glorot_normal',
        #                    kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv3)
        # deconv3 = KL.BatchNormalization()(deconv3)
        # deconv3 = KL.Activation('relu')(deconv3)
        # deconv3 = KL.Conv2D(128, (3, 3), padding='same',
        #                    kernel_initializer='glorot_normal',
        #                    kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv3)
        # deconv3 = KL.BatchNormalization()(deconv3)
        # deconv3 = KL.Activation('relu')(deconv3)

        deconv4 = KL.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid',
                                      kernel_initializer='glorot_normal',
                                      kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(conv5)
        deconv4 = KL.Activation('relu')(deconv4)
        deconv4 = KL.Concatenate(axis=3)([conv4 , deconv4])
        deconv4 = KL.Conv2D(64, (3, 3), padding='same',
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv4)
        deconv4 = KL.BatchNormalization()(deconv4)
        deconv4 = KL.Activation('relu')(deconv4)
        deconv4 = KL.Conv2D(64, (3, 3), padding='same',
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv4)
        deconv4 = KL.BatchNormalization()(deconv4)
        deconv4 = KL.Activation('relu')(deconv4)

        deconv5 = KL.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='valid',
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv4)
        deconv5 = KL.Activation('relu')(deconv5)
        deconv5 = KL.Concatenate(axis=3)([conv3, deconv5])
        deconv5 = KL.Conv2D(32, (3, 3), padding='same',
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv5)
        deconv5 = KL.BatchNormalization()(deconv5)
        deconv5 = KL.Activation('relu')(deconv5)
        deconv5 = KL.Conv2D(32, (3, 3), padding='same',
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv5)
        deconv5 = KL.BatchNormalization()(deconv5)
        deconv5 = KL.Activation('relu')(deconv5)

        deconv6 = KL.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='valid',
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv5)
        deconv6 = KL.Activation('relu')(deconv6)
        deconv6 = KL.Concatenate(axis=3)([conv2, deconv6])
        deconv6 = KL.Conv2D(16, (3, 3), padding='same',
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv6)
        deconv6 = KL.BatchNormalization()(deconv6)
        deconv6 = KL.Activation('relu')(deconv6)
        deconv6 = KL.Conv2D(16, (3, 3), padding='same',
                            kernel_initializer='glorot_normal',
                            kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv6)
        deconv6 = KL.BatchNormalization()(deconv6)
        deconv6= KL.Activation('relu')(deconv6)

        deconv7 = KL.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='valid',
                                     kernel_initializer='glorot_normal',
                                     kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv6)
        deconv7 = KL.Activation('relu')(deconv7)
        deconv7 = KL.Concatenate(axis=3)([conv1, deconv7])
        deconv7 = KL.Conv2D(8, (3, 3), padding='same',
                           kernel_initializer='glorot_normal',
                           kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv7)
        deconv7 = KL.BatchNormalization()(deconv7)
        deconv7 = KL.Activation('relu')(deconv7)
        deconv7 = KL.Conv2D(8, (3, 3), padding='same',
                           kernel_initializer='glorot_normal',
                           kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv7)
        deconv7 = KL.BatchNormalization()(deconv7)
        deconv7 = KL.Activation('relu')(deconv7)
        print(deconv7.get_shape())

        classify = KL.Conv2D(self.config.NUM_CLASSES, (1, 1),
                             activation='sigmoid')(deconv7)
        # classify = KL.Conv2D(self.config.NUM_CLASSES, (1, 1),
        #                      kernel_initializer='glorot_normal',
        #                      kernel_regularizer=KR.l2(self.WEIGHT_DECAY))(deconv7)

        model = KM.Model(inputs=inputs, outputs=classify)

        return model

    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def compile(self, learning_rate, momentum):
        # optimizer = keras.optimizers.RMSprop(lr=learning_rate)
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        self.model.compile(optimizer=optimizer, loss=losses.binary_crossentropy,
                           metrics=[self.dice_coeff])

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": "input_concat4",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE,
                                       augment=False)

        # Callbacks
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                              factor=0.1,
                                              patience=4,
                                              verbose=1,
                                              epsilon=1e-4),

            keras.callbacks.ModelCheckpoint(monitor='val_loss',
                                            filepath=self.config.get_weights_name(),
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True),

            keras.callbacks.TensorBoard(log_dir='logs',
                                        histogram_freq=0, write_graph=True, write_images=False)]

        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Train
        self.model.fit_generator(
            generator=train_generator,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            workers=max(self.config.BATCH_SIZE // 2, 2),
            use_multiprocessing=True,
        )

    def dice(self, y_true, y_pred):
        smooth = 1.
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = (y_true_f * y_pred_f).sum()
        score = (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
        return score

    def _fast_hist(self, label_pred, label_true, num_classes):
        mask = (label_true >= 0) & (label_true < num_classes)
        hist = np.bincount(
            num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    def compute_IOU(self, pred, gt, num_classes=2):
        hist = np.zeros((num_classes, num_classes))
        hist += self._fast_hist(pred.flatten(), gt.flatten(), num_classes)
        dice = self.dice(y_true=gt, y_pred=pred)

        # axis 0: gt, axis 1: prediction
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        # mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        return acc, acc_cls, iu, fwavacc, dice

    def evaluation(self, dataset, filepath):
        image_ids = dataset.image_ids
        threshold = 0.5
        iu = np.zeros(len(image_ids))

        for index, image_id in enumerate(image_ids):
            # Load image
            image, gt_mask = load_image_gt(dataset, self.config, image_id)
            image = mold_image(image.astype(np.float32), self.config)

            pred = self.model.predict(np.array([image]), verbose=0)
            # pred = np.reshape(pred, (pred.shape[1], pred.shape[2]))
            pred[pred < threshold] = 0
            pred[pred > threshold] = 1
            _, _, iou, _, dice = self.compute_IOU(pred.astype(np.int64), gt_mask.astype(np.int64))
            iu[index] = iou[1]

            info = dataset.image_info[image_id]
            filename = info['filename']

            # filename = filename + '_' + str(iu[index])[0:5]
            # utils.save_results(os.path.join(filepath, filename + '.png'),
            #                    (np.reshape(pred, (pred.shape[1], pred.shape[2])) * 255).astype(np.uint8))

            # utils.save_results(os.path.join(filepath, filename + '.png'),
            #                    (np.reshape(pred, (pred.shape[1], pred.shape[2]))).astype(np.uint8))

            print (filename, iu[index], dice, pred.sum())

        return iu

    def save_box(self, dataset, box_dir):
        isExists = os.path.exists(box_dir)
        if not isExists:
            os.makedirs(box_dir)

        val_box_info = open('/media/Disk/wangfuyu/Mask_RCNN/crop_results/box_info.txt', 'r')
        lines = val_box_info.readlines()
        box_dict = {}

        for line in lines:
            tmp = re.split(' ', line)
            filename = tmp[0]
            y1 = int(tmp[1])
            x1 = int(tmp[2])
            y2 = int(tmp[3])
            x2 = int(tmp[4])

            box_dict[filename] = [y1, x1, y2, x2]

        image_ids = dataset.image_ids
        threshold = 0.5

        for index, image_id in enumerate(image_ids):
            # Load image
            image, gt_mask = load_image_gt(dataset, self.config, image_id)
            image = mold_image(image.astype(np.float32), self.config)

            info = dataset.image_info[image_id]
            filename = info['filename']
            y1, x1, y2, x2 = box_dict[filename]
            # print(y1, y2, y1 + y2)

            pred = self.model.predict(np.array([image]), verbose=0)
            pred = np.reshape(pred, (pred.shape[1], pred.shape[2]))
            # print (pred.shape)
            pred = scipy.misc.imresize(pred, (y2 - y1, x2 - x1), interp='bilinear').astype(np.float32) / 255.0
            pred[pred < threshold] = 0
            pred[pred > threshold] = 1
            # pred.squeeze(axis=2)

            result = np.zeros((1024, 1024))
            result[y1: y2, x1: x2] = pred

            cv2.imwrite(os.path.join(box_dir, filename + '.png'),
                        (result*255).astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

    def save_results(self, dataset, save_dir):
        image_ids = dataset.image_ids
        threshold = 0.5

        for index, image_id in enumerate(image_ids):
            # Load image
            image, gt_mask = load_image_gt(dataset, self.config, image_id)
            image = mold_image(image.astype(np.float32), self.config)

            pred = self.model.predict(np.array([image]), verbose=0)
            # print (pred.shape)
            pred = np.reshape(pred, (pred.shape[1], pred.shape[2]))
            pred[pred < threshold] = 0
            pred[pred > threshold] = 1

            info = dataset.image_info[image_id]
            filename = info['filename']

            # filename = filename + '_' + str(iu[index])[0:5]
            # utils.save_results(os.path.join(filepath, filename + '.png'),
            #                    (np.reshape(pred, (pred.shape[1], pred.shape[2])) * 255).astype(np.uint8))

            cv2.imwrite(os.path.join(save_dir, filename + '.png'),
                        pred.astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            # cv2.imwrite(os.path.join(save_dir, filename + '.png'),
            #             (pred * 255).astype(np.uint8), [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
