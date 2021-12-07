import tensorflow.compat.v2 as tf

class TunableGaussianNoise(tf.keras.layers.Layer):
    """
    Настраиваемый слой аддитивного гауссова шума.
    """
    def __init__(self, stddev, **kwargs):
        super(TunableGaussianNoise, self).__init__()
        self._name = kwargs['name']
        self.enabled = tf.Variable(initial_value=True, trainable=False)
        self.stddev = tf.Variable(initial_value=stddev, trainable=False)

    def call(self, inputs):
        if self.enabled:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=tf.float32)
            return inputs + noise
        else:
            return inputs

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
