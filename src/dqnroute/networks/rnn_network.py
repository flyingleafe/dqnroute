import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .q_network import QNetwork
from .input_type_networks import *
from .activations import get_activation

def get_rnn_cell(cell_type, csize, act_foo):
    if cell_type == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(num_units=csize, activation=act_foo)
    elif cell_type == 'gru':
        return tf.contrib.rnn.GRUCell(num_units=csize, activation=act_foo)
    else:
        raise Exception('Unknown RNN cell type: ' + cell_type)

class RQNetwork(QNetwork):
    def getHiddenLayers(self, ff_layers=[64], rnn_size=64, cell_type='lstm',
                        ff_activation='relu', rnn_activation='tanh', scope='', **kwargs):
        scope_pref = scope + '_' if scope != '' else ''
        self.lstm_layers_label = scope_pref + cell_type + '_' + ff_activation + '_' + \
                                 '_'.join(map(str, ff_layers)) + '-' + \
                                 rnn_activation + '_' + str(rnn_size)
        self.cell_type = cell_type
        self.rnn_size = rnn_size

        self.resetRnnState()

        scope = self.getLabel()
        n = self.graph_size
        ff_act_foo = get_activation(ff_activation)
        rnn_act_foo = get_activation(rnn_activation)

        dense = self.all_inps
        for (i, lsize) in enumerate(ff_layers):
            dense = slim.fully_connected(inputs=dense, num_outputs=lsize, activation_fn=ff_act_foo,
                                         scope=scope+'_dense'+str(i))

        # Placeholders for batch size / episode length
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.episode_length = tf.placeholder(dtype=tf.int32)

        rnn_input = tf.reshape(dense, shape=(self.batch_size, self.episode_length, rnn_size))
        rnn_cell = get_rnn_cell(cell_type, rnn_size, rnn_act_foo)
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

        self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(inputs=rnn_input, cell=rnn_cell,
                                                         dtype=tf.float32, initial_state=self.state_in,
                                                         scope=scope+'_rnn')

        self.rnn_out = tf.reshape(self.rnn_out, shape=(-1, rnn_size))
        return slim.fully_connected(inputs=self.rnn_out, num_outputs=n, activation_fn=None, scope=scope+'_dense_out')

    def _get_rand_vec(self, batch_size, random_init_state):
        if random_init_state:
            return np.random.rand(batch_size, self.rnn_size) * 2 - 1
        return np.zeros((batch_size, self.rnn_size))

    def resetRnnState(self, batch_size=1, random_init_state=False):
        if self.cell_type == 'lstm':
            self.rnn_state_val = (self._get_rand_vec(batch_size, random_init_state),
                                  self._get_rand_vec(batch_size, random_init_state))
        elif self.cell_type == 'gru':
            self.rnn_state_val = self._get_rand_vec(batch_size, random_init_state)
        else:
            raise Exception('etogo ne mojet byt')

    def getLabel(self):
        lbl = super().getLabel()
        return lbl + '_' + self.lstm_layers_label

    def _determineEpLen(self, x, batch_size):
        inp_len = len(x['dst'])
        if inp_len % batch_size != 0:
            raise Exception('Cannot determine episode length!')
        return inp_len / batch_size

    def mkFeedDict(self, x, y=None, batch_size=1):
        fdict = super().mkFeedDict(x, y)
        episode_length = self._determineEpLen(x, batch_size)
        fdict[self.batch_size] = batch_size
        fdict[self.episode_length] = episode_length
        fdict[self.state_in] = self.rnn_state_val
        return fdict

    def fit(self, session, x, y, batch_size=1, save_old_state=False, outer_state=None, **kwargs):
        old_rnn_state = self.rnn_state_val
        if outer_state is not None:
            self.rnn_state_val = outer_state
        loss, _, new_state = session.run([self.loss, self.updateModel, self.rnn_state],
                                         self.mkFeedDict(x, y, batch_size))
        if save_old_state:
            self.rnn_state_val = old_rnn_state
        else:
            self.rnn_state_val = new_state
        return loss

    def predict(self, session, x, batch_size=1, outer_state=None):
        if outer_state is not None:
            self.rnn_state_val = outer_state
        out, new_state = session.run([self.Qout, self.rnn_state], self.mkFeedDict(x, batch_size=batch_size))
        self.rnn_state_val = new_state
        if outer_state is not None:
            return out, new_state
        return out

    def preTrainOneEpoch(self, session, data, batch_size=1, random_init_state=False,
                         **kwargs):
        sum_loss = 0
        loss_cnt = 0
        for (batch, targets) in self.preparePreTrainBatches(data, batch_size=batch_size, **kwargs):
            loss = self.fit(session, batch, targets, batch_size=batch_size, **kwargs)
            self.resetRnnState(batch_size, random_init_state=random_init_state)
            sum_loss += loss
            loss_cnt += 1
        self.save(session) # because rnns are slow
        return sum_loss / loss_cnt

    def preparePreTrainBatches(self, data, batch_size=1, episode_length=32,
                               episode_col='addr', shuffle_eps=False, **kwargs):
        ep_ids = data[episode_col].unique()
        target_cols = get_target_cols(self.graph_size)

        if shuffle_eps:
            random.shuffle(ep_ids)

        for ep_id in ep_ids:
            episode = data[data[episode_col] == ep_id].sort_values('time')
            count = episode.shape[0]
            for (a, b) in make_batches(count, episode_length):
                mini_episode = episode[a:b]
                yield (self.makeInputFromData(mini_episode), mini_episode[target_cols].values)

class RQNetworkAmatrix(RQNetwork, AMatrixInputNetwork):
    pass

class RQNetworkAmatrixTriangle(RQNetwork, AMatrixTriangleInputNetwork):
    pass

class RQNetworkAddFlatAmatrix(RQNetwork, FlatAdditionalInput, AMatrixInputNetwork):
    pass

class RQNetworkAddFlatAmatrixTriangle(RQNetwork, FlatAdditionalInput, AMatrixTriangleInputNetwork):
    pass
