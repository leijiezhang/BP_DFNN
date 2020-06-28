from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
from keras.engine import InputSpec
from keras import backend as K


class RuleGaussianLayer(Layer):

    def __init__(self, n_rules):
        super(RuleGaussianLayer, self).__init__()
        self.n_rules = n_rules
        self.mu = 0
        self.sigma = 0
        # self.w = 0

    def build(self, input_shape):  # Create the state of the layer (weights)
        mu_init = tf.random_normal_initializer()
        self.mu = tf.Variable(
            initial_value=mu_init(shape=input_shape,
                                 dtype='float32'),
            trainable=True)
        # mu_init = tf.random_normal_initializer()
        # self.mu = tf.Variable(
        #     initial_value=mu_init(shape=input_shape,
        #                           dtype='float32'),
        #     trainable=True)
        b_init = tf.ones_initializer()
        self.sigma = tf.Variable(
            initial_value=b_init(shape=input_shape, dtype='float32'),
            trainable=True)

    def call(self, inputs, **kwargs):  # Defines the computation from inputs to outputs
        # 1) fuzzification layer (Gaussian membership functions)
        inputs = tf.expand_dims(inputs, axis=1)

        layer_fuzzify = tf.exp(
            -(tf.broadcast_to(inputs, [1, self.n_rules, 1]) - self.mu) ** 2 / (2 * self.sigma ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = K.prod(layer_fuzzify, axis=2)
        # layer_rule[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-8)).double()
        layer_rule = layer_rule + np.asarray(10 ** (-198), dtype=np.float)
        # 3) normalization layer (normalize firing strength)
        inv_frn = tf.expand_dims(tf.math.reciprocal(K.sum(layer_rule, axis=1)), axis=1)
        layer_normalize = tf.math.multiply(layer_rule, tf.broadcast_to(inv_frn, layer_rule.shape))
        return layer_normalize


class BottomMultiLayer(Layer):

    def __init__(self, n_rules):
        super(BottomMultiLayer, self).__init__()

    def build(self, input_shape):  # Create the state of the layer (weights)
        mu_init = tf.random_normal_initializer()
        self.mu = tf.Variable(
            initial_value=mu_init(shape=input_shape,
                                 dtype='float32'),
            trainable=True)
        # mu_init = tf.random_normal_initializer()
        # self.mu = tf.Variable(
        #     initial_value=mu_init(shape=input_shape,
        #                           dtype='float32'),
        #     trainable=True)
        b_init = tf.ones_initializer()
        self.sigma = tf.Variable(
            initial_value=b_init(shape=input_shape, dtype='float32'),
            trainable=True)

    def call(self, inputs, **kwargs):  # Defines the computation from inputs to outputs
        # 1) fuzzification layer (Gaussian membership functions)
        inputs = tf.expand_dims(inputs, axis=1)

        layer_fuzzify = tf.exp(
            -(tf.broadcast_to(inputs, [1, self.n_rules, 1]) - self.mu) ** 2 / (2 * self.sigma ** 2))
        # 2) rule layer (compute firing strength values)
        layer_rule = K.prod(layer_fuzzify, axis=2)
        # layer_rule[layer_rule < 10 ^ (-8)] = torch.tensor(10 ^ (-8)).double()
        layer_rule = layer_rule + np.asarray(10 ** (-198), dtype=np.float)
        # 3) normalization layer (normalize firing strength)
        inv_frn = tf.expand_dims(tf.math.reciprocal(K.sum(layer_rule, axis=1)), axis=1)
        layer_normalize = tf.math.multiply(layer_rule, tf.broadcast_to(inv_frn, layer_rule.shape))
        return layer_normalize
