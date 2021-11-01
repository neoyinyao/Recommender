import tensorflow as tf
from tensorflow import keras
from layers import InterestExtract, InterestEvolve, Attention, MLP


class DIEN(keras.Model):
    def __init__(self, item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size,
                 interest_extract_gru_units, interest_evolve_gru_units, mlp_units):
        super().__init__()
        self.item_embedding_layer = keras.layers.Embedding(item_vocab_size, item_embedding_size)
        self.cat_embedding_layer = keras.layers.Embedding(cat_vocab_size, cat_embedding_size)
        self.interest_extract_layer = InterestExtract(interest_extract_gru_units)
        self.attention = Attention(item_embedding_size + cat_embedding_size)
        self.interest_evolve = InterestEvolve(interest_evolve_gru_units)
        self.mlp = MLP(mlp_units, last_activation=tf.nn.sigmoid)

    def compute_embedding(self, item, cat):
        item_embedding = self.item_embedding_layer(item)
        cat_embedding = self.cat_embedding_layer(cat)
        return tf.concat([item_embedding, cat_embedding], axis=-1)

    def call(self, inputs):
        query_item, query_cat, pos_history_items, pos_history_cat, neg_history_items, neg_history_cat, mask = inputs
        query_embedding = self.compute_embedding(query_item, query_cat)
        pos_history_embedding = self.compute_embedding(pos_history_items, pos_history_cat)
        neg_history_embedding = self.compute_embedding(neg_history_items, neg_history_cat)
        hidden_state, auxiliary_loss = self.interest_extract_layer(pos_history_embedding, neg_history_embedding, mask)
        attention_score = self.attention(query_embedding, hidden_state, mask)  # None,his_len,1
        final_state = self.interest_evolve(hidden_state, attention_score, mask)
        query_embedding = tf.squeeze(query_embedding, axis=1)
        inputs = tf.concat([query_embedding, final_state], axis=-1)
        prob = self.mlp(inputs)
        return prob, auxiliary_loss
