import tensorflow as tf
from tensorflow import keras
from layers import LocalActivationUnit, MLP


class DIN(keras.Model):
    def __init__(self, item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size, mlp_units, max_length):
        super().__init__()
        self.mlp = MLP(mlp_units, tf.nn.sigmoid)
        self.item_embedding = keras.layers.Embedding(item_vocab_size, item_embedding_size)
        self.cat_embedding = keras.layers.Embedding(cat_vocab_size, cat_embedding_size)
        self.local_activation_unit = LocalActivationUnit(max_length)

    def compute_embedding(self, item, cat):
        item_embedding = self.item_embedding(item)
        cat_embedding = self.cat_embedding(cat)
        embedding = tf.concat([item_embedding, cat_embedding], axis=-1)
        return embedding

    def call(self, inputs):
        query_item, query_cat, his_item, his_cat, mask = inputs
        query_embedding = self.compute_embedding(query_item, query_cat)
        his_embedding = self.compute_embedding(his_item, his_cat)
        history_representation = self.local_activation_unit(query_embedding, his_embedding, mask)
        history_representation = tf.squeeze(history_representation, axis=1)
        query_embedding = tf.squeeze(query_embedding, axis=1)
        embedding = tf.concat(
            [query_embedding, history_representation], axis=-1)
        prob = self.mlp(embedding)
        return prob
