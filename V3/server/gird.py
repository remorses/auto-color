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


with tf.device(device_A):

    ipa = tf.placeholder(dtype=tf.float32, shape=(None, 1))
    ip1 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1))
    ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    ip4 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 4))
    ip3x = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))

    girder = load_model('girder.net')
    gird_op = (1 - girder([1 - ip1 / 255.0, ip4, 1 - ip3 / 255.0])) * 255.0




session.run(tf.global_variables_initializer())


girder.load_weights('girder.net')




def go_gird(sketch, latent, hint):
    return session.run(gird_op, feed_dict={
        ip1: sketch[None, :, :, None], ip3: latent[None, :, :, :], ip4: hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)

