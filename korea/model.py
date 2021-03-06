import os
import sys
import random
import re
import logging
import numpy as np
import cv2
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

    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = utils.resize_mask(mask, scale, padding)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

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
                outputs = []

                yield inputs, outputs

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


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   dilation_rate=(1, 1), use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), dilation_rate=dilation_rate,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  dilation_rate=dilation_rate, name=conv_name_base + '2b',
                  use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)

    shortcut = KL.Conv2D(nb_filter2, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


class Net():
    def __init__(self, mode, config, model_dir):
        self.mode = mode
        self.config = config
        self.WEIGHT_DECAY = config.WEIGHT_DECAY
        if config.concat:
            self.input_shape = (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 4)
        else:
            self.input_shape = (self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 3)

        self.model_dir = model_dir
        self.set_log_dir()

        self.model = self.build()

    def log_layers(self):
        layers = self.model.layers
        for layer in layers:
            print(layer.name)

    def build(self):
        input_image = KL.Input(shape=self.input_shape, name='input_image')

        x = KL.ZeroPadding2D((2, 2))(input_image)
        x = KL.Conv2D(32, (5, 5), name='conv1', use_bias=True)(x)
        x = KL.BatchNormalization(axis=3, name='bn_conv1')(x)
        x = KL.Activation('relu')(x)
        x = KL.Conv2D(32, (3, 3), padding='same', name='conv2', use_bias=True)(x)
        x = KL.BatchNormalization(axis=3, name='bn_conv2')(x)
        x = KL.Activation('relu')(x)

        x = conv_block(x, 3, [32, 32], stage=3, block='a')
        x = identity_block(x, 3, [32, 32], stage=3, block='b')
        x = identity_block(x, 3, [32, 32], stage=3, block='c')

        x = conv_block(x, 3, [32, 32], stage=4, block='a')
        x = identity_block(x, 3, [32, 32], stage=4, block='b', dilation_rate=(3, 3))
        x = identity_block(x, 3, [32, 32], stage=4, block='c', dilation_rate=(3, 3))

        classify = KL.Conv2D(2, (1, 1), name='classifier')(x)
        classify = KL.Lambda(lambda t: tf.image.resize_images(
            t, [self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM]))(classify)

        if self.mode == 'train':
            input_mask = KL.Input(shape=[self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM, 1],
                                  name='input_mask')
            loss = KL.Lambda(lambda x: losses.sparse_categorical_crossentropy(*x), name="loss")(
                [input_mask, classify])

            inputs = [input_image, input_mask]
            outputs = [classify, loss]
        else:
            inputs = input_image
            outputs = classify

        model = KM.Model(inputs=inputs, outputs=outputs)

        return model

    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return score

    def set_log_dir(self):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        self.log_dir = os.path.join(self.model_dir, "stage1")

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "korea_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        print (filepath)
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
        loss_names = ["loss"]
        for name in loss_names:
            layer = self.model.get_layer(name)
            if layer.output in self.model.losses:
                continue
            self.model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
                      for w in self.model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.model.compile(optimizer=optimizer, loss=[
                                 None] * len(self.model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.model.metrics_names:
                continue
            layer = self.model.get_layer(name)
            self.model.metrics_names.append(name)
            self.model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))

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

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Train
        self.model.fit_generator(
            generator=train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            workers=max(self.config.BATCH_SIZE // 2, 2),
            use_multiprocessing=True,
        )

        self.epoch = max(self.epoch, epochs)

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
        # threshold = 0.5
        iu = np.zeros(len(image_ids))
        record_txt = open('/media/Disk/wangfuyu/Mask_RCNN/korea/record.txt', 'w')
        record_txt.write('filename' + ' ' + 'iu' + '\n')

        images_dir = '/media/Disk/wangfuyu/Mask_RCNN/korea/256/images'
        masks_dir = '/media/Disk/wangfuyu/Mask_RCNN/korea/256/masks'

        isExists = os.path.exists(images_dir)
        if not isExists:
            os.makedirs(images_dir)
        isExists = os.path.exists(masks_dir)
        if not isExists:
            os.makedirs(masks_dir)

        for index, image_id in enumerate(image_ids):
            # Load image
            image, gt_mask = load_image_gt(dataset, self.config, image_id)

            info = dataset.image_info[image_id]
            filename = info['filename']
            cv2.imwrite(os.path.join(images_dir, filename + '.jpg'), image,
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
            cv2.imwrite(os.path.join(masks_dir, filename + '.png'), (gt_mask*255).astype(np.uint8),
                        [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

            '''
            image = mold_image(image.astype(np.float32), self.config)

            pred = self.model.predict(np.array([image]), verbose=0)

            print (pred.shape)
            pred = np.argmax(pred, axis=3)
            print (pred.shape)
            pred = np.squeeze(pred, axis=0)
            print (pred.shape)
            pred = np.expand_dims(pred, axis=2)

            print (pred.shape)

            # pred[pred < threshold] = 0
            # pred[pred > threshold] = 1
            _, _, iou, _, dice = self.compute_IOU(pred.astype(np.int64), gt_mask.astype(np.int64))
            iu[index] = iou[1]

            info = dataset.image_info[image_id]
            filename = info['filename']

            utils.save_results(os.path.join(filepath, filename + '.png'),
                               (pred.squeeze(axis=2) * 255).astype(np.uint8))

            record_txt.write(filename + ' ' + str(iu[index])[0:8] + '\n')

            # utils.save_results(os.path.join(filepath, filename + '.png'),
            #                    (np.reshape(pred, (pred.shape[1], pred.shape[2]))).astype(np.uint8))

            print (filename, iu[index], dice, pred.sum())
            '''

        return iu


