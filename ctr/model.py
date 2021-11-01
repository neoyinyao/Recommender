from layers import MLP, DotInteraction
import tensorflow as tf
from tensorflow import keras


class DeepFM(keras.Model):
    def __init__(self, embedding_size, vocab_size, num_int_fea, num_cat_fea, mlp_units):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding_layer = keras.layers.Embedding(vocab_size, embedding_size)
        self.num_int_fea = num_int_fea
        self.num_cat_fea = num_cat_fea
        self.mlp = MLP(mlp_units, final_activation=None)

    def call(self, inputs, training=None, mask=None):
        cat_features, int_features = inputs['cat_features'], inputs['int_features']
        int_features = tf.reshape(int_features, (-1, self.num_int_fea))
        cat_features = tf.reshape(cat_features, (-1, self.num_cat_fea))
        cat_embedding = self.embedding_layer(cat_features)

        sum_square = tf.square(tf.reduce_sum(cat_embedding, axis=1))
        square_sum = tf.reduce_sum(tf.square(cat_embedding), axis=1)
        interaction = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)

        deep_cat_input = tf.reshape(cat_embedding, (-1, self.num_cat_fea * self.embedding_size))
        deep_input = tf.concat([deep_cat_input, int_features], axis=1)
        dense_output = self.mlp(deep_input)
        dense_output = tf.squeeze(dense_output, axis=1)
        output = interaction + dense_output
        output = tf.nn.sigmoid(output)
        return output


class DLRM(keras.Model):
    def __init__(self, bottom_mlp_units, top_mlp_units, embedding_size, vocab_size, num_cat_fea, num_int_fea):
        super().__init__()
        self.num_cat_fea = num_cat_fea
        self.num_int_fea = num_int_fea
        self.bottom_mlp = MLP(bottom_mlp_units, final_activation='relu')
        self.top_mlp = MLP(top_mlp_units, final_activation='sigmoid')
        self.embedding_size = embedding_size
        self.embedding_layer = keras.layers.Embedding(vocab_size, embedding_size)
        self.interaction = DotInteraction(False, True)

    def call(self, x):
        cat_features, int_features = x['cat_features'], x['int_features']
        int_features = tf.reshape(int_features, (-1, self.num_int_fea))
        cat_features = tf.reshape(cat_features, (-1, self.num_cat_fea))
        cat_embedding = self.embedding_layer(cat_features)
        bmlp_activation = self.bottom_mlp(int_features)
        bmlp_trandform = tf.expand_dims(bmlp_activation, axis=1)
        concat_feature = tf.concat([cat_embedding, bmlp_trandform], axis=1)
        interaction = self.interaction(concat_feature)
        tmlp_input = tf.concat([interaction, bmlp_activation], axis=1)
        tmlp_input = tf.reshape(tmlp_input, (-1, (self.num_cat_fea + 1) ** 2 + self.embedding_size))
        output = self.top_mlp(tmlp_input)
        output = tf.squeeze(output, 1)
        return output
