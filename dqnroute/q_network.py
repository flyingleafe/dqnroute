import tensorflow as tf
import tensorflow.contrib.slim as slim

from keras.layers import *

class Qnetwork():
    def __init__(self, n, rnn_cell, myScope):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        self.neighbors_input = tf.placeholder(shape=(None,n),dtype=tf.float32)
        self.addr_input = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.dst_input = tf.placeholder(shape=(None,n), dtype=tf.float32)
        self.amatrix_input = tf.placeholder(shape=(None, n*n), dtype=tf.float32)

        #self.trainLength = tf.placeholder(dtype=tf.int32)
        #We take the output from the final convolutional layer and send it to a recurrent layer.
        #The input must be reshaped into [batch x trace x units] for rnn processing, 
        #and then returned to [batch x units] when sent through the upper levles.
        #self.batch_size = tf.placeholder(dtype=tf.int32)
        #self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])
        #self.state_in = cell.zero_state(self.batch_size, tf.float32)
        #self.rnn,self.rnn_state = tf.nn.dynamic_rnn(\
        #        inputs=self.convFlat,cell=rnn_cell,dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
        #self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        #The output from the recurrent player is then split into separate Value and Advantage streams

        self.all_inps = tf.concat([self.neighbors_input, self.addr_input, self.dst_input, self.amatrix_input], 1)
        layer1 = slim.fully_connected(inputs=self.all_inps, num_outputs=64, scope=myScope+'_dense1')

        #self.train_length = tf.placeholder(dtype=tf.int32)
        #self.batch_size = tf.placeholder(dtype=tf.int32)
        #self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=la,cell=rnn_cell,
        #                                             dtype=tf.float32,initial_state=self.state_in,scope=myScope+'_rnn')
        layer2 = slim.fully_connected(inputs=layer1, num_outputs=64, scope=myScope+'_dense2')
        self.dense_out = slim.fully_connected(inputs=layer2, num_outputs=n, activation_fn=None, scope=myScope+'_dense_out')

        lambda_l = Lambda(lambda x: (1 - x) * -1000000)(self.neighbors_input)
        self.Qout = tf.add(self.dense_out, lambda_l)


#         self.streamA,self.streamV = tf.split(self.rnn,2,1)
#         self.AW = tf.Variable(tf.random_normal([h_size//2,4]))
#         self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
#         self.Advantage = tf.matmul(self.streamA,self.AW)
#         self.Value = tf.matmul(self.streamV,self.VW)

#         self.salience = tf.gradients(self.Advantage,self.imageIn)
#         #Then combine them together to get our final Q-values.
#         self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        #self.predict = tf.argmax(self.Qout,1)

        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        #self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        #self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        #self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)

        #self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        #self.td_error = tf.square(self.targetQ - self.Q)

        #In order to only propogate accurate gradients through the network, we will mask the first
        #half of the losses for each trace as per Lample & Chatlot 2016
        #self.maskA = tf.zeros([self.batch_size,self.trainLength//2])
        #self.maskB = tf.ones([self.batch_size,self.trainLength//2])
        #self.mask = tf.concat([self.maskA,self.maskB],1)
        #self.mask = tf.reshape(self.mask,[-1])
        #self.loss = tf.reduce_mean(self.td_error * self.mask)
        self.target = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target, self.Qout)

        self.trainer = tf.train.RMSPropOptimizer(0.001)
        self.updateModel = self.trainer.minimize(self.loss)
