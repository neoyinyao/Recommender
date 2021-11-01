import tensorflow as tf
from tensorflow import keras


def compute_history_sum(his_embedding, mask):
    """
    :param his_embedding: None,his_len,embedding_size
    :param mask: None,his_len
    :return:
    """
    mask = tf.expand_dims(mask, axis=-1)
    his_embedding *= mask
    his_embedding_sum = tf.reduce_sum(his_embedding, axis=1)
    return his_embedding_sum


def compute_history_average(his_embedding, mask):
    """
    :param his_embedding: None,his_len,embedding_size
    :param mask: None,his_len
    :return:
    """
    mask = tf.expand_dims(mask, axis=-1)
    his_embedding *= mask
    mask_sum = tf.reduce_sum(mask, axis=1)  # None,1
    his_embedding_average = tf.reduce_sum(his_embedding, axis=1) / mask_sum
    return his_embedding_average


class MLP(keras.layers.Layer):
    def __init__(self, units, last_activation):
        super().__init__()
        self.mlp = [keras.layers.Dense(unit, activation='relu') for unit in units[:-1]]
        self.mlp.append(keras.layers.Dense(units[-1], activation=last_activation))

    def call(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class LocalActivationUnit(tf.keras.layers.Layer):
    def __init__(self, history_max_length):
        super().__init__()
        self.history_max_length = history_max_length
        self.layer_1 = tf.keras.layers.Dense(80, activation=tf.nn.sigmoid)
        self.layer_2 = tf.keras.layers.Dense(40, activation=tf.nn.sigmoid)
        self.layer_3 = tf.keras.layers.Dense(1)

    def call(self, query, history, mask):
        """
        :param query: None,1,embedding_size
        :param history: None,history_max_length,embedding_size
        :param mask: None,history_max_length
        :return:
        """
        query = tf.repeat(query, repeats=[self.history_max_length], axis=1)  # None,history_max_length,embedding_size
        concat = tf.concat([query, history, query - history, query * history], axis=-1)
        weights = self.layer_1(concat)
        weights = self.layer_2(weights)
        weights = self.layer_3(weights)  # None,history_max_length,1
        mask = tf.expand_dims(mask, axis=-1)  # None,history_max_length,1
        weights *= mask
        history_representation = tf.matmul(weights, history, transpose_a=True)  # None,1,embedding_size
        return history_representation


class InterestExtract(keras.layers.Layer):
    def __init__(self, gru_units, fc_units=[100, 50, 1]):
        super().__init__()
        self.gru = keras.layers.GRU(units=gru_units, return_sequences=True)
        self.fc = MLP(fc_units, last_activation='sigmoid')

    def compute_logit(self, hidden_state, history, mask):
        history = tf.boolean_mask(history, mask)
        concat = tf.concat([hidden_state, history], axis=-1)
        concat = self.fc(concat)
        concat = tf.squeeze(concat, axis=-1)
        return concat

    def compute_auxiliary_loss(self, hidden_state, pos_history, neg_history, mask):
        hidden_state = hidden_state[:, :-1, :]
        pos_history = pos_history[:, 1:, :]
        neg_history = neg_history[:, 1:, :]
        mask = mask[:, 1:]
        # hidden_state_mask = tf.boolean_mask(hidden_state, mask)
        # pos_logit = self.compute_logit(hidden_state_mask, pos_history, mask)
        # neg_logit = self.compute_logit(hidden_state_mask, neg_history, mask)
        # label_sum = tf.reduce_sum(mask)
        # pos_label = tf.ones(shape=(label_sum,))
        # neg_label = tf.zeros(shape=(label_sum,))
        # pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(pos_label, pos_logit)
        # neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(neg_label, neg_logit)
        # auxiliary_loss = tf.reduce_mean(pos_loss + neg_loss)
        # return auxiliary_loss
        mask = tf.cast(mask, tf.float32)  # None,his_len
        click_input_ = tf.concat([hidden_state, pos_history], -1)  # None,his_len,embedding_size
        noclick_input_ = tf.concat([hidden_state, neg_history], -1)  # None,his_len,embedding_size
        click_prop_ = self.fc(click_input_)
        click_prop_ = tf.squeeze(click_prop_, axis=-1)  # None,his_len
        noclick_prop_ = self.fc(noclick_input_)
        noclick_prop_ = tf.squeeze(noclick_prop_, axis=-1)  # None,his_len
        click_loss_ = - tf.math.log(click_prop_) * mask
        noclick_loss_ = - tf.math.log(1.0 - noclick_prop_) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def call(self, pos_history, neg_history, mask):
        """
        pos_his: None,his_len,input_size
        neg_his: None,his_len,input_size
        mask: None,his_len
        """
        hidden_state = self.gru(pos_history, mask=mask)
        auxiliary_loss = self.compute_auxiliary_loss(hidden_state, pos_history, neg_history, mask)
        return hidden_state, auxiliary_loss


class Attention(keras.layers.Layer):
    def __init__(self, input_size):
        super().__init__()
        self.fc = keras.layers.Dense(input_size, activation=None)

    def call(self, query, hidden_state, mask):
        """
        hidden_state: None,his_len,hidden_size
        query: None,1,input_size
        mask: None,his_len
        """
        trans = self.fc(hidden_state)  # None,his_len,input_size
        score = tf.matmul(trans, query, transpose_b=True)  # None,his_len,1
        mask = tf.expand_dims(mask, axis=-1)
        score += (1 - mask) * -1e9
        score = tf.nn.softmax(score, axis=1)
        return score


class AUGRUCell(keras.layers.AbstractRNNCell):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.update_gate = keras.layers.Dense(units, activation=tf.nn.sigmoid)
        self.reset_gate = keras.layers.Dense(units, activation=tf.nn.sigmoid)
        self.hidden_layer = keras.layers.Dense(units, activation=tf.nn.tanh)

    @property
    def state_size(self):
        return self.units

    def call(self, inputs, states):
        """
        states: None,hidden_size
        inputs: None,input_size
        """
        prev_output = states[0]
        attention_score = inputs[:, -1:]
        inputs = inputs[:, :-1]
        concat = tf.concat([prev_output, inputs], axis=-1)
        update_state = self.update_gate(concat)
        reset_state = self.reset_gate(concat)

        hidden_state = self.hidden_layer(tf.concat([inputs, reset_state * prev_output], axis=-1))
        update_state *= attention_score
        output = update_state * hidden_state + (1 - update_state) * prev_output
        return output, output


class InterestEvolve(keras.layers.Layer):
    def __init__(self, gru_units):
        super().__init__()
        self.augru = keras.layers.RNN(AUGRUCell(gru_units))

    def call(self, history_state, attention_score, mask):
        """
        history_state: None,his_len,input_size
        attention_score: None,his_len,1
        """
        inputs = tf.concat([attention_score, history_state], axis=-1)
        output = self.augru(inputs, mask=mask)
        return output
