import tensorflow as tf
from tensorflow import keras


def compute_his_average(his_embedding, mask):
    """
    :param his_embedding: None,his_len,embedding_size
    :param mask: None,his_len
    :return:his_embedding_average
    """
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.cast(mask, dtype=his_embedding.dtype)
    his_embedding *= mask
    mask_sum = tf.reduce_sum(mask, axis=1)  # None,1
    embedding_sum = tf.reduce_sum(his_embedding, axis=1)
    his_embedding_average = embedding_sum / mask_sum
    return his_embedding_average


class MLP(keras.layers.Layer):
    def __init__(self, units, last_activation):
        super().__init__()
        self.bn = keras.layers.BatchNormalization()
        self.mlp = [keras.layers.Dense(unit, activation='relu') for unit in units[:-1]]
        self.mlp.append(keras.layers.Dense(units[-1], activation=last_activation))

    def call(self, inputs, training=False):
        inputs = self.bn(inputs, training=training)
        for layer in self.mlp:
            inputs = layer(inputs)
        return inputs


class LocalActivationUnit(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layer_1 = tf.keras.layers.Dense(80, activation=tf.nn.sigmoid)
        self.layer_2 = tf.keras.layers.Dense(40, activation=tf.nn.sigmoid)
        self.layer_3 = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        """
        :param target: None,1,embedding_size
        :param history: None,history_max_length,embedding_size
        :param mask: None,history_max_length
        :return:
        """
        target, history = inputs
        target = tf.repeat(target, repeats=[history.shape[1]], axis=1)  # None,his_max_length,embedding_size
        concat = tf.concat([target, history, target - history, target * history], axis=-1)
        weights = self.layer_1(concat)
        weights = self.layer_2(weights)
        weights = self.layer_3(weights)  # None,history_max_length,1
        mask = tf.expand_dims(mask, axis=-1)  # None,history_max_length,1
        mask = tf.cast(mask, dtype=weights.dtype)
        weights *= mask
        history_representation = tf.matmul(weights, history, transpose_a=True)  # None,1,embedding_size
        history_representation = tf.squeeze(history_representation, axis=1)
        return history_representation


class AuxiliaryNet(keras.layers.Layer):
    def __init__(self, mlp_units):
        super(AuxiliaryNet, self).__init__()
        # self.bn = keras.layers.BatchNormalization()
        self.layers = [keras.layers.Dense(unit, activation='sigmoid') for unit in mlp_units[:-1]]
        self.layers.append(keras.layers.Dense(mlp_units[-1], activation=None))

    def call(self, inputs, training=False, **kwargs):
        # inputs = self.bn(inputs, training=training)
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class InterestExtract(keras.layers.Layer):
    def __init__(self, gru_units):
        super().__init__()
        self.gru = keras.layers.GRU(units=gru_units, return_sequences=True)
        self.auxiliary_net = AuxiliaryNet([80, 40, 1])

    def compute_logits(self, inputs, training=False):
        hidden_state, history = inputs
        concat = tf.concat([hidden_state, history], axis=-1)
        logits = self.auxiliary_net(concat, training=training)
        logits = tf.squeeze(logits, axis=-1)
        return logits

    def compute_auxiliary_loss(self, inputs, training=False, mask=None):
        hidden_state, pos_his_embedding, neg_his_embedding = inputs
        hidden_state = hidden_state[:, :-1, :]
        pos_history = pos_his_embedding[:, 1:, :]
        neg_history = neg_his_embedding[:, 1:, :]
        mask = mask[:, 1:]
        pos_logits = self.compute_logits((hidden_state, pos_history), training)  # None,his_len
        neg_logits = self.compute_logits((hidden_state, neg_history), training)  # None,his_len
        pos_label = tf.ones_like(pos_logits)
        neg_label = tf.zeros_like(neg_logits)
        pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(pos_label, pos_logits)  # None,his_len
        neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(neg_label, neg_logits)  # None,his_len
        mask = tf.cast(mask, pos_loss.dtype)
        pos_loss *= mask
        neg_loss *= mask
        auxiliary_loss_concat = tf.concat([pos_loss, neg_loss], axis=-1)
        auxiliary_loss_sum = tf.reduce_sum(auxiliary_loss_concat, axis=-1)  # None,
        mask_sum = tf.reduce_sum(mask, axis=-1) * 2.
        auxiliary_loss = auxiliary_loss_sum / mask_sum
        return auxiliary_loss  # None,
        # https://github.com/mouna99/dien.git implementation
        # click_input_ = tf.concat([hidden_state, pos_history], -1)  # None,his_len,embedding_size
        # noclick_input_ = tf.concat([hidden_state, neg_history], -1)  # None,his_len,embedding_size
        # click_prop_ = self.mlp(click_input_, training)
        # click_prop_ = tf.sigmoid(click_prop_)
        # click_prop_ = tf.squeeze(click_prop_, axis=-1)  # None,his_len
        # noclick_prop_ = self.mlp(noclick_input_, training)
        # noclick_prop_ = tf.sigmoid(noclick_prop_)
        # noclick_prop_ = tf.squeeze(noclick_prop_, axis=-1)  # None,his_len
        # click_loss_ = - tf.math.log(click_prop_) * mask
        # noclick_loss_ = - tf.math.log(1.0 - noclick_prop_) * mask
        # loss_ = click_loss_ + noclick_loss_
        # # loss_ = tf.reduce_mean(loss_)
        # return loss_

    def call(self, inputs, training=False, mask=None):
        """
        pos_his: None,his_len,input_size
        neg_his: None,his_len,input_size
        mask: None,his_len
        """
        pos_history, neg_history = inputs
        hidden_state = self.gru(pos_history, mask=mask)
        auxiliary_loss = self.compute_auxiliary_loss((hidden_state, pos_history, neg_history), training, mask)
        return hidden_state, auxiliary_loss


class DIENAttention(keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        target_shape = input_shape[0]
        hidden_shape = input_shape[1]
        self.kernel = self.add_weight(shape=(hidden_shape[-1], target_shape[-1]))

    def call(self, inputs, training=False, mask=None):
        """
        hidden_state: None,his_len,hidden_size
        target: None,1,input_size
        mask: None,his_len
        """
        target, hidden_state = inputs
        trans = tf.matmul(hidden_state, self.kernel)  # None,his_len,input_size
        score = tf.matmul(trans, target, transpose_b=True)  # None,his_len,1
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(mask, dtype=score.dtype)
        score += (1. - mask) * -1e9
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

    def call(self, inputs, training=False, mask=None):
        """
        history_state: None,his_len,input_size
        attention_score: None,his_len,1
        """
        history_state, attention_score = inputs
        inputs = tf.concat([history_state, attention_score], axis=-1)
        output = self.augru(inputs, mask=mask)
        return output
