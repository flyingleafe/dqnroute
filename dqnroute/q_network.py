import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from keras.layers import *

class Qnetwork():
    def __init__(self, n, rnn_cell, myScope):
        self.neighbors_input = tf.placeholder(shape=(None,n),dtype=tf.float32)
        self.addr_input = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.dst_input = tf.placeholder(shape=(None,n), dtype=tf.float32)
        # self.out_links_input = tf.placeholder(shape=(None,n), dtype=tf.float32)
        self.amatrix_input = tf.placeholder(shape=(None, n*n), dtype=tf.float32)

        self.all_inps = tf.concat([self.neighbors_input, self.addr_input, self.dst_input,
                                   self.amatrix_input], 1)
                                   # self.out_links_input, self.amatrix_input], 1)
        layer1 = slim.fully_connected(inputs=self.all_inps, num_outputs=64, scope=myScope+'_dense1')

        layer2 = slim.fully_connected(inputs=layer1, num_outputs=64, scope=myScope+'_dense2')
        self.dense_out = slim.fully_connected(inputs=layer2, num_outputs=n, activation_fn=None, scope=myScope+'_dense_out')

        lambda_l = Lambda(lambda x: (1 - tf.minimum(x, 1))*-1000000)(self.neighbors_input)
        self.Qout = tf.add(self.dense_out, lambda_l)

        self.target = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target, self.Qout)

        self.trainer = tf.train.RMSPropOptimizer(0.001)
        self.updateModel = self.trainer.minimize(self.loss)
