import tensorflow as tf
import numpy as np

from keras.layers import Lambda

from ..utils import *
from ..constants import TF_MODELS_DIR
from .optimizers import get_optimizer

class QNetwork:
    def __init__(self, n, optimizer='rmsprop', optimizer_params={}, **kwargs):
        self.graph_size = n

        opt_label = optimizer if type(optimizer) == str else 'custom'
        self.label = 'dqn_' + opt_label

        self.neighbors_input = tf.placeholder(shape=(None,n),dtype=tf.float32)
        self.addr_input = tf.placeholder(shape=(None,), dtype=tf.int32)
        self.dst_input = tf.placeholder(shape=(None,), dtype=tf.int32)

        self.addr_onehot = tf.one_hot(self.addr_input, n)
        self.dst_onehot = tf.one_hot(self.dst_input, n)

        add_input = self.getAdditionalInput(**kwargs)
        inp_ls = [self.neighbors_input, self.addr_onehot, self.dst_onehot] + add_input

        self.all_inps = tf.concat(inp_ls, 1)

        self.hidden_out = self.getHiddenLayers(**kwargs)

        lambda_l = Lambda(lambda x: (1 - tf.minimum(x, 1))*-1000000)(self.neighbors_input)
        self.Qout = tf.add(self.hidden_out, lambda_l)

        self.target = tf.placeholder(shape=(None, n), dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.target, self.Qout)

        self.trainer = get_optimizer(optimizer, optimizer_params)
        self.updateModel = self.trainer.minimize(self.loss)
        self.saver = tf.train.Saver()

    def getLabel(self):
        return self.label

    def getSavePath(self):
        return TF_MODELS_DIR + '/' + self.getLabel()

    def getAdditionalInput(self, **kwargs):
        return []

    def makeInputFromData(self, data):
        n = self.graph_size
        return {
            'neighbors': data[get_neighbors_cols(n)].values,
            'addr': data['addr'].values,
            'dst': data['dst'].values
        }

    def getHiddenLayers(self, **kwargs):
        pass

    def mkFeedDict(self, x, y=None):
        feed_dict = {
            self.neighbors_input:x['neighbors'],
            self.addr_input:x['addr'],
            self.dst_input:x['dst']
        }
        if y is not None:
            feed_dict[self.target] = y
        return feed_dict

    def fit(self, session, x, y, **kwargs):
        loss, _ = session.run([self.loss, self.updateModel], self.mkFeedDict(x, y))
        return loss

    def predict(self, session, x, **kwargs):
        return session.run(self.Qout, self.mkFeedDict(x))

    def save(self, session):
        return self.saver.save(session, self.getSavePath())

    def restore(self, session):
        return self.saver.restore(session, self.getSavePath())

    def preTrain(self, session, data, epochs=1, **kwargs):
        epochs_losses = []
        for i in range(epochs):
            print('Epoch {}... '.format(i), end='')
            avg_loss = self.preTrainOneEpoch(session, data, **kwargs)
            print('loss: {}'.format(avg_loss))
            epochs_losses.append(avg_loss)
        return epochs_losses

    def preTrainOneEpoch(self, session, data, **kwargs):
        sum_loss = 0
        loss_cnt = 0
        for (batch, targets) in self.preparePreTrainBatches(data, **kwargs):
            loss = self.fit(session, batch, targets, **kwargs)
            sum_loss += loss
            loss_cnt += 1
        return sum_loss / loss_cnt

    def preparePreTrainBatches(self, data, batch_size=32, **kwargs):
        count = data.shape[0]
        target_cols = get_target_cols(self.graph_size)
        for (a, b) in make_batches(count, batch_size):
            batch = data[a:b]
            yield (self.makeInputFromData(batch), batch[target_cols].values)
