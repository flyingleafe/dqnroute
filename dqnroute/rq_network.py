import tensorflow as tf
import tensorflow.contrib.slim as slim

from keras.layers import Lambda

class RQnetwork():
    def __init__(self, n, hsize, rnn_cell, myScope):
        self.neighbors_input = tf.placeholder(shape=(None,n),dtype=tf.float32)
        self.addr_input = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.dst_input = tf.placeholder(shape=(None,n), dtype=tf.float32)
        self.amatrix_input = tf.placeholder(shape=(None, n*n), dtype=tf.float32)

        self.all_inps = tf.concat([self.neighbors_input, self.addr_input, self.dst_input, self.amatrix_input], 1)
        layer1 = slim.fully_connected(inputs=self.all_inps, num_outputs=hsize, scope=myScope+'_dense1')

        self.train_length = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32)

        # Reshape input to the sequence
        self.rnn_input = tf.reshape(layer1, shape=[self.batch_size, self.train_length, hsize])

        self.hidden_state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=rnn_cell, dtype=tf.float32,
                                                     initial_state=self.hidden_state_in, scope=myScope+'_rnn')
        self.rnn_output = tf.reshape(self.rnn, shape=[-1, hsize])

        self.dense_out = slim.fully_connected(inputs=self.rnn_output, num_outputs=n,
                                              activation_fn=None, scope=myScope+'_dense_out')

        lambda_l = Lambda(lambda x: (1 - x) * -1000000)(self.neighbors_input)
        self.Qout = tf.add(self.dense_out, lambda_l)

        self.target = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target, self.Qout)

        self.trainer = tf.train.RMSPropOptimizer(0.001)
        self.updateModel = self.trainer.minimize(self.loss)
