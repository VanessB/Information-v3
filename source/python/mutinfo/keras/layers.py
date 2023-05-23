import tensorflow.compat.v2 as tf

class TunableGaussianNoise(tf.keras.layers.Layer):
    """
    Tunable layer for additive Gaussian noise.
    """
    
    def __init__(self, stddev, **kwargs):
        super(TunableGaussianNoise, self).__init__()
        self._name = kwargs['name']
        self.enabled = tf.Variable(initial_value=True, trainable=False)
        self.stddev = tf.Variable(initial_value=stddev, trainable=False)

    def call(self, inputs):
        return tf.cond(self.enabled, lambda: inputs + tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=tf.float32), lambda: inputs)

    def get_config(self):
        config = super(TunableGaussianNoise, self).get_config().copy()
        config.update({
            'enabled': self.enabled.value().numpy(),
            'stddev': self.stddev.value().numpy(),
        })
        return config

    def set_config(self, config):
        super(TunableGaussianNoise, self).set_config(config)
        self.enabled.assign(config['enabled'])
        self.stddev.assign(config['stddev'])
