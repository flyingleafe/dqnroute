import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .q_network import QNetwork
from .input_type_networks import *
from .activations import get_activation

class FFNetwork(QNetwork):
    def getHiddenLayers(self, layers=[64, 64], scope='', activation='relu', **kwargs):
        scope_pref = scope + '_' if scope != '' else ''
        self.ff_layers_label = scope_pref + 'ff_' + activation + '_' + '_'.join(map(str, layers))
        scope = self.getLabel()
        n = self.graph_size
        act_foo = get_activation(activation)

        dense = self.all_inps
        for (i, lsize) in enumerate(layers):
            dense = slim.fully_connected(inputs=dense, num_outputs=lsize, activation_fn=act_foo,
                                         scope=scope+'_dense'+str(i))
        return slim.fully_connected(inputs=dense, num_outputs=n, activation_fn=None, scope=scope+'_dense_out')

    def getLabel(self):
        lbl = super().getLabel()
        return lbl + '_' + self.ff_layers_label

class FFNetworkAmatrix(FFNetwork, AMatrixInputNetwork):
    pass

class FFNetworkAmatrixTriangle(FFNetwork, AMatrixTriangleInputNetwork):
    pass

class FFNetworkAddFlatAmatrix(FFNetwork, FlatAdditionalInput, AMatrixInputNetwork):
    pass

class FFNetworkAddFlatAmatrixTriangle(FFNetwork, FlatAdditionalInput, AMatrixTriangleInputNetwork):
    pass
