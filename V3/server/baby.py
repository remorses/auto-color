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

    baby = load_model('baby.net')
    baby_place = tf.concat([- 512 * tf.ones_like(ip4[:, :, :, 3:4]), 128 * tf.ones_like(ip4[:, :, :, 3:4]), 128 * tf.ones_like(ip4[:, :, :, 3:4])], axis=3)
    baby_yuv = RGB2YUV(ip4[:, :, :, 0:3])
    baby_alpha = tf.where(x=tf.zeros_like(ip4[:, :, :, 3:4]), y=tf.ones_like(ip4[:, :, :, 3:4]), condition=tf.less(ip4[:, :, :, 3:4], 128))
    baby_hint = baby_alpha * baby_yuv + (1 - baby_alpha) * baby_place
    baby_op = YUV2RGB(baby(tf.concat([ip1, baby_hint], axis=3)))



session.run(tf.global_variables_initializer())


baby.load_weights('baby.net')




def go_baby(sketch, local_hint):
    return session.run(baby_op, feed_dict={
        ip1: sketch[None, :, :, None], ip4: local_hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)

