import tensorflow as tf
import keras
import numpy as np
from config import *
from keras.models import load_model


def ToGray(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    return 0.30 * R + 0.59 * G + 0.11 * B


def RGB2YUV(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = 0.492 * (B - Y) + 128
    V = 0.877 * (R - Y) + 128
    return tf.concat([Y, U, V], axis=3)


def YUV2RGB(x):
    Y = x[:, :, :, 0:1]
    U = x[:, :, :, 1:2]
    V = x[:, :, :, 2:3]
    R = Y + 1.140 * (V - 128)
    G = Y - 0.394 * (U - 128) - 0.581 * (V - 128)
    B = Y + 2.032 * (U - 128)
    return tf.concat([R, G, B], axis=3)


def VGG2RGB(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1]


session = keras.backend.get_session()



with tf.device(device_B):

    ip3B = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

    tail = load_model('models/tail.net')
    pads = 7
    tail_op = tail(tf.pad(ip3B / 255.0, [[0, 0], [pads, pads], [pads, pads], [0, 0]], 'REFLECT'))[:, pads*2:-pads*2, pads*2:-pads*2, :] * 255.0


session.run(tf.global_variables_initializer())


tail.load_weights('models/tail.net')





def go_tail(x):
    return session.run(tail_op, feed_dict={
        ip3B: x[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


