import tensorflow as tf
import numpy as np

from .q_network import QNetwork
from ..utils import *

class AMatrixInputNetwork(QNetwork):
    def getAdditionalInput(self, **kwargs):
        add_inp = super().getAdditionalInput(**kwargs)
        self.amatrix_input = tf.placeholder(shape=(None, self.graph_size*self.graph_size), dtype=tf.float32)
        return [self.amatrix_input] + add_inp

    def makeInputFromData(self, data):
        inps = super().makeInputFromData(data)
        n = self.graph_size
        inps['amatrix'] = data[get_amatrix_cols(n)].values
        return inps

    def mkFeedDict(self, x, y=None):
        fdict = super().mkFeedDict(x, y)
        fdict[self.amatrix_input] = x['amatrix']
        return fdict

    def getLabel(self):
        lbl = super().getLabel()
        return lbl + '_' + 'amatrix'

class AMatrixTriangleInputNetwork(QNetwork):
    def getAdditionalInput(self, **kwargs):
        add_inp = super().getAdditionalInput(**kwargs)
        n = self.graph_size
        self.amatrix_input = tf.placeholder(shape=(None, n*(n-1)/2), dtype=tf.float32)
        return [self.amatrix_input] + add_inp

    def makeInputFromData(self, data):
        inps = super().makeInputFromData(data)
        n = self.graph_size
        inps['amatrix'] = data[get_amatrix_triangle_cols(n)].values
        return inps

    def mkFeedDict(self, x, y=None):
        fdict = super().mkFeedDict(x, y)
        fdict[self.amatrix_input] = x['amatrix']
        return fdict

    def getLabel(self):
        lbl = super().getLabel()
        return lbl + '_' + 'amatrix_tr'

class FlatAdditionalInput(QNetwork):
    def getAdditionalInput(self, flat_inputs={}, **kwargs):
        add_inp = super().getAdditionalInput(**kwargs)
        n = self.graph_size
        self.flat_add_inps_dims = {}
        self.flat_add_inputs = {}
        for (name, dim) in flat_inputs.items():
            m = dim if dim != 'graph_size' else n
            self.flat_add_inps_dims[name] = m
            self.flat_add_inputs[name] = tf.placeholder(shape=(None, m), dtype=tf.float32)
        return list(self.flat_add_inputs.values()) + add_inp

    def makeInputFromData(self, data):
        inps = super().makeInputFromData(data)
        for (name, dim) in self.flat_add_inps_dims.items():
            inps[name] = data[mk_num_list(name+'_', dim)].values
        return inps

    def mkFeedDict(self, x, y=None):
        fdict = super().mkFeedDict(x, y)
        for (name, plh) in self.flat_add_inputs.items():
            fdict[plh] = x[name]
        return fdict

    def getLabel(self):
        lbl = super().getLabel()
        return lbl + '_' + '_'.join([name + str(dim) for (name, dim) in
                                     self.flat_add_inps_dims.items()])
