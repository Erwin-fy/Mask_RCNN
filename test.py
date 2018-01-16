import numpy as np
import tensorflow as tf
import cv2
import scipy.misc

a = np.array([[[1,2],[2,3],[3,4],[4,5]], [[1,2],[2,3],[3,4],[4,5]], [[1,2],[2,3],[3,4],[4,5]]])
a = a.transpose()
a = np.reshape(a, (8, 3))
print a
# for i, level in enumerate(range(2, 6)):
#
#     ix = tf.where(tf.equal([[2,3,4,5,6]], level))
#     print ix, ix.shape
#     box_indices = tf.cast(ix[:, 0], tf.int32)
#     print box_indices, box_indices.get_shape()
#
#     ix = np.where(np.equal([[1,2,3,4,5]], level))
#     print ix
#     # box_indices = np.cast(ix[:, 0], np.int32)
#     # print box_indices