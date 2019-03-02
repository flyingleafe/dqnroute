import tensorflow as tf

def get_optimizer(name, params):
    if type(name) != str:
        return name
    if name == 'rmsprop':
        ps = {'learning_rate': 0.001}
        ps.update(params)
        return tf.train.RMSPropOptimizer(**ps)
    elif name == 'adam':
        return tf.train.AdamOptimizer(**params)
    elif name == 'adadelta':
        return tf.train.AdadeltaOptimizer(**params)
    elif name == 'adagrad':
        ps = {'learning_rate': 0.001}
        ps.update(params)
        return tf.train.AdagradOptimizer(**ps)
