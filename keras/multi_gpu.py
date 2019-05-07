from __future__ import print_function
import tensorflow.python.keras
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import Input, Lambda
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import concatenate
import tensorflow as tf

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
session = tf.Session(config=session_config)


def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part * L:]
    return x[part * L:(part + 1) * L]


def to_multi_gpu(model, n_gpus=2):
    if n_gpus == 1:
        return model

    with tf.device('/cpu:0'):
        x = Input(model.input_shape[1:])
    towers = []
    for g in range(n_gpus):
        with tf.device('/gpu:' + str(g)):
            slice_g = Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus': n_gpus, 'part': g})(x)
            towers.append(model(slice_g))

    with tf.device('/cpu:0'):
        # Deprecated
        # merged = merge(towers, mode='concat', concat_axis=0)
        merged = concatenate(towers,axis=0)
    return Model(inputs=[x], outputs=merged)
