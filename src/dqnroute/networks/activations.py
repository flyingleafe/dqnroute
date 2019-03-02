import tensorflow as tf

def get_activation(name):
    if type(name) != str:
        return name
    if name == 'relu':
        return tf.nn.relu
    elif name == 'tanh':
        return tf.tanh
    elif name == 'sigmoid':
        return tf.sigmoid
    else:
        raise Exception('Unknown activation function: ' + name)
