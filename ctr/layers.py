import tensorflow as tf
from tensorflow import keras


class MLP(keras.layers.Layer):
    def __init__(self, units, final_activation):
        super().__init__()
        self.mlp = [keras.layers.Dense(unit) for unit in units[:-1]]
        self.mlp.append(keras.layers.Dense(units[-1], activation=final_activation))

    def call(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class DotInteraction(keras.layers.Layer):
    def __init__(self, self_interaction, skip_gather):
        self.self_interaction = self_interaction
        self.skip_gather = skip_gather
        super().__init__()

    def call(self, inputs):
        feature_dim = tf.shape(inputs)[1]
        xmatrix = tf.matmul(inputs, inputs, transpose_b=True)
        ones = tf.ones_like(xmatrix)
        if self.self_interaction:
            upper_matrix = tf.linalg.band_part(ones, -1, 0)
            lower_matrix = ones - upper_matrix
            output_shape = feature_dim * (feature_dim + 1) // 2
        else:
            lower_matrix = tf.linalg.band_part(ones, -1, 0)
            upper_matrix = ones - lower_matrix
            output_shape = feature_dim * (feature_dim - 1) // 2
        if self.skip_gather:
            condition = tf.cast(upper_matrix, tf.bool)
            activation = tf.where(condition, xmatrix, tf.zeros_like(xmatrix))
            activation = tf.reshape(activation, (-1, feature_dim * feature_dim))
        else:
            mask = tf.cast(upper_matrix, tf.bool)
            activation = tf.boolean_mask(xmatrix, mask)
            activation = tf.reshape(activation, (-1, output_shape))
        return activation
