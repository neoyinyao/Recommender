import tensorflow as tf
from tensorflow import keras

from layers import MLP


class ESMM(keras.Model):
    def __init__(self, mlp_units, feat_vocab, embedding_size):
        super().__init__()
        self.embedding_layer = {feat: keras.layers.Embedding(vocab_size, embedding_size) for feat, vocab_size in
                                feat_vocab.items()}
        self.ctr = MLP(mlp_units, tf.nn.sigmoid)
        self.cvr = MLP(mlp_units, tf.nn.sigmoid)

    def compute_embedding(self, inputs):
        embedding = [self.embedding_layer[feat](inputs[feat]) for feat in inputs]
        embedding = tf.concat(embedding, axis=-1)
        embedding = tf.squeeze(embedding, axis=1)
        return embedding

    def call(self, inputs, training=None, mask=None):
        embedding = self.compute_embedding(inputs)
        p_ctr = self.ctr(embedding)
        p_cvr = self.cvr(embedding)
        p_ctcvr = p_cvr * p_ctr
        outputs = tf.concat([p_ctr, p_ctcvr], axis=-1)
        return outputs

    def compute_cvr(self, inputs):
        embedding = self.compute_embedding(inputs)
        p_cvr = self.cvr(embedding)
        return p_cvr

    def compute_ctr(self, inputs):
        embedding = self.compute_embedding(inputs)
        p_ctr = self.ctr(embedding)
        return p_ctr

    def compute_ctcvr(self, inputs):
        embedding = self.compute_embedding(inputs)
        p_cvr = self.cvr(embedding)
        p_ctr = self.ctr(embedding)
        p_ctcvr = p_cvr * p_ctr
        return p_ctcvr
