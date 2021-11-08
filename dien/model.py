import tensorflow as tf
from tensorflow import keras

from layers import MLP, compute_his_average, LocalActivationUnit, InterestEvolve, InterestExtract, DIENAttention


class BaseModel(keras.Model):
    def __init__(self, item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size, mlp_units):
        super().__init__()
        self.mlp = MLP(mlp_units, 'sigmoid')
        self.item_embedding = keras.layers.Embedding(item_vocab_size, item_embedding_size, mask_zero=True)
        self.cat_embedding = keras.layers.Embedding(cat_vocab_size, cat_embedding_size, mask_zero=True)

    def compute_flat_embedding(self, inputs):
        item, cat = inputs
        item_embedding = self.item_embedding(item)
        cat_embedding = self.cat_embedding(cat)
        embedding = tf.concat([item_embedding, cat_embedding], axis=-1)
        return embedding

    def compute_prob(self, inputs):
        return self.call(inputs)

    def call(self, inputs, training=False, mask=None):
        mask = self.item_embedding.compute_mask(inputs['pos_his_item'])  # (None,max_hi_len) tf.bool
        target_embedding = self.compute_flat_embedding(
            (inputs['target_item'], inputs['target_cat']))  # None,1,embedding_size
        target_embedding = tf.squeeze(target_embedding, axis=1)  # None,embedding_size
        his_embedding = self.compute_flat_embedding(
            (inputs['pos_his_item'], inputs['pos_his_cat']))  # None,his_len,embedding_size
        history_representation = compute_his_average(his_embedding, mask)  # None,embedding_size
        embedding = tf.concat([target_embedding, history_representation], axis=-1)
        prob = self.mlp(embedding, training=training)
        return prob


class DIN(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.local_activation_unit = LocalActivationUnit()

    def call(self, inputs, training=False, mask=None):
        mask = self.item_embedding.compute_mask(inputs['pos_his_item'])
        target_embedding = self.compute_flat_embedding(
            (inputs['target_item'], inputs['target_cat']))  # None,1,embedding_size
        his_embedding = self.compute_flat_embedding(
            (inputs['pos_his_item'], inputs['pos_his_cat']))  # None,his_len,embedding_size
        history_representation = self.local_activation_unit((target_embedding, his_embedding), training=training,
                                                            mask=mask)
        target_embedding = tf.squeeze(target_embedding, axis=1)
        embedding = tf.concat([target_embedding, history_representation], axis=-1)
        prob = self.mlp(embedding, training=training)
        return prob


class DIEN(BaseModel):
    def __init__(self, interest_extract_gru_units, interest_evolve_gru_units, **kwargs):
        super().__init__(**kwargs)
        self.interest_extract_layer = InterestExtract(interest_extract_gru_units)
        self.attention = DIENAttention()
        self.interest_evolve = InterestEvolve(interest_evolve_gru_units)

    def compute_prob(self, inputs):
        prob, _ = self.call(inputs)
        return prob

    def call(self, inputs, training=False, mask=None):
        mask = self.item_embedding.compute_mask(inputs['pos_his_item'])
        target_embedding = self.compute_flat_embedding((inputs['target_item'], inputs['target_cat']))
        pos_history_embedding = self.compute_flat_embedding((inputs['pos_his_item'], inputs['pos_his_cat']))
        neg_history_embedding = self.compute_flat_embedding((inputs['neg_his_item'], inputs['neg_his_cat']))

        hidden_state, auxiliary_loss = self.interest_extract_layer((pos_history_embedding, neg_history_embedding),
                                                                   training, mask)
        attention_score = self.attention((target_embedding, hidden_state), training, mask)  # None,his_len,1
        history_representation = self.interest_evolve((hidden_state, attention_score), training, mask)
        target_embedding = tf.squeeze(target_embedding, axis=1)
        embedding = tf.concat([target_embedding, history_representation], axis=-1)
        prob = self.mlp(embedding)
        return prob, auxiliary_loss
