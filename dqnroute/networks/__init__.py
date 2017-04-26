from .optimizers import *
from .q_network import *
from .input_type_networks import *
from .ff_network import *
from .rnn_network import *

def get_qnetwork_class(label):
    if label == 'ff_none':
        return FFNetwork
    elif label == 'ff_amatrix':
        return FFNetworkAmatrix
    elif label == 'ff_amatrix_triangle':
        return FFNetworkAmatrixTriangle
    elif label == 'ff_amatrix_flat_add':
        return FFNetworkAddFlatAmatrix
    elif label == 'ff_amatrix_tr_flat_add':
        return FFNetworkAddFlatAmatrixTriangle
    elif label == 'rnn_none':
        return RQNetwork
    elif label == 'rnn_amatrix':
        return RQNetworkAmatrix
    elif label == 'rnn_amatrix_triangle':
        return RQNetworkAmatrixTriangle
    elif label == 'rnn_amatrix_flat_add':
        return RQNetworkAddFlatAmatrix
    elif label == 'rnn_amatrix_tr_flat_add':
        return RQNetworkAddFlatAmatrixTriangle
    else:
        raise Exception('Unknown Q-network type provided!')

