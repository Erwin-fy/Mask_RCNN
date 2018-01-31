import numpy as np
import tensorflow as tf
import cv2
import scipy.misc

a = np.array([[[1,2],[3,4],[5,6],[7,8]], [[9,10],[11,12],[13,14],[15,16]], [[17,18],[19,20],[21,22],[23,24]]])
a.resize((2,2,2))
print a
cv2.dra
# a = np.reshape(a, (8, 3))
# print a
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