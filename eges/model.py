import tensorflow as tf
from tensorflow import keras


class DeepWalk(keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.input_embedding = keras.layers.Embedding(vocab_size, embedding_size)
        self.output_embedding = keras.layers.Embedding(vocab_size, embedding_size)

    def call(self, inputs):
        """
        :param inputs: (query,pos+neg)
        :return: None,1+num_neg
        """
        query, match = inputs
        hidden = self.input_embedding(query)  # None,1,embedding_size
        match_embedding = self.output_embedding(match)  # None,1+num_neg,embedding_size
        logits = tf.matmul(match_embedding, hidden, transpose_b=True)  # None,1+num_neg,1
        logits = tf.squeeze(logits, axis=-1)
        return logits

    def get_hidden(self, inputs):
        hidden = self.input_embedding(inputs)
        return hidden


class GES(keras.Model):
    def __init__(self, id_vocab_size, cat_vocab_size, brand_vocab_size, embedding_size):
        super().__init__()
        self.id_embedding = keras.layers.Embedding(id_vocab_size, embedding_size)
        self.cat_embedding = keras.layers.Embedding(cat_vocab_size, embedding_size)
        self.brand_embedding = keras.layers.Embedding(brand_vocab_size, embedding_size)
        self.output_embedding = keras.layers.Embedding(id_vocab_size, embedding_size)

    def call(self, inputs):
        query_item_id, query_cat_id, query_brand_id, match = inputs
        hidden = self.get_hidden((query_item_id, query_cat_id, query_brand_id))
        match_embedding = self.output_embedding(match)  # None,1+num_neg,embedding_size
        logits = tf.matmul(match_embedding, hidden, transpose_b=True)  # None,1+num_neg,1
        logits = tf.squeeze(logits, axis=-1)
        return logits

    def get_hidden(self, inputs):
        query_item_id, query_cat_id, query_brand_id = inputs
        query_id_embedding = self.id_embedding(query_item_id)
        query_cat_embedding = self.cat_embedding(query_cat_id)
        query_brand_embedding = self.brand_embedding(query_brand_id)
        hidden = (query_id_embedding + query_cat_embedding + query_brand_embedding) / 3
        return hidden


class EGES(keras.Model):
    def __init__(self, id_vocab_size, cat_vocab_size, brand_vocab_size, embedding_size, num_side):
        super().__init__()
        self.id_embedding = keras.layers.Embedding(id_vocab_size, embedding_size)
        self.cat_embedding = keras.layers.Embedding(cat_vocab_size, embedding_size)
        self.brand_embedding = keras.layers.Embedding(brand_vocab_size, embedding_size)
        self.output_embedding = keras.layers.Embedding(id_vocab_size, embedding_size)
        self.weight_embedding = keras.layers.Embedding(id_vocab_size, num_side)

    def call(self, inputs):
        """
        inputs: (query,id:pos+neg,cat:pos+neg,brand:pos+neg,labels)
        None,1+num_neg
        """
        query_item_id, query_cat_id, query_brand_id, match = inputs
        hidden = self.get_hidden((query_item_id, query_cat_id, query_brand_id))
        match_embedding = self.output_embedding(match)  # None,1+num_neg,embedding_size
        logits = tf.matmul(match_embedding, hidden, transpose_b=True)  # None,1+num_neg,1
        logits = tf.squeeze(logits, axis=-1)
        return logits

    def get_hidden(self, inputs):
        query_item_id, query_cat_id, query_brand_id = inputs
        query_id_embedding = self.id_embedding(query_item_id)
        query_cat_embedding = self.cat_embedding(query_cat_id)
        query_brand_embedding = self.brand_embedding(query_brand_id)
        query_embedding_concat = tf.concat((query_id_embedding, query_cat_embedding, query_brand_embedding),
                                           axis=1)  # None,num_side,embedding_size
        weights = self.weight_embedding(query_item_id)  # None,1,num_side
        weights = tf.nn.softmax(weights, axis=-1)
        hidden = tf.matmul(weights, query_embedding_concat)
        return hidden
