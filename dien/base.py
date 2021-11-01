import tensorflow as tf
from tensorflow import keras
from layers import MLP, compute_history_average


class BaseModel(keras.Model):
    def __init__(self, item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size, mlp_units):
        super().__init__()
        self.mlp = MLP(mlp_units, 'sigmoid')
        self.item_embedding = keras.layers.Embedding(item_vocab_size, item_embedding_size)
        self.cat_embedding = keras.layers.Embedding(cat_vocab_size, cat_embedding_size)

    def compute_embedding(self, item, cat):
        item_embedding = self.item_embedding(item)
        cat_embedding = self.cat_embedding(cat)
        embedding = tf.concat([item_embedding, cat_embedding], axis=-1)
        return embedding

    def call(self, inputs):
        """
        :param inputs:
        mask: None,his_len
        :return:
        """
        query_item, query_cat, his_item, his_cat, mask = inputs
        query_embedding = self.compute_embedding(query_item, query_cat)
        his_embedding = self.compute_embedding(his_item, his_cat)  # None,his_len,embedding_size
        his_average = compute_history_average(his_embedding, mask)
        query_embedding = tf.squeeze(query_embedding, axis=1)
        embedding = tf.concat([query_embedding, his_average], axis=-1)
        prob = self.mlp(embedding)
        return prob
