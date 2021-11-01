import tensorflow as tf
from tensorflow import keras

from layers import MLP


class BaseModel(keras.Model):
    def __init__(self, hidden_units, last_activation, feat_vocab, embedding_size):
        super().__init__()
        self.mlp = MLP(hidden_units, last_activation)
        self.embedding_layer = {feat: keras.layers.Embedding(count, embedding_size) for feat, count in
                                feat_vocab.items()}

    def call(self, inputs, training=None, mask=None):
        embedding = [self.embedding_layer[feat](inputs[feat]) for feat in inputs]
        embedding = tf.concat(embedding, axis=-1)
        embedding = tf.squeeze(embedding, axis=1)
        prob = self.mlp(embedding)
        return prob
