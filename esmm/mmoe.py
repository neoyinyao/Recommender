from layers import MLP

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class MMOE(keras.Model):
    def __init__(self, num_tasks, num_experts, expert_hidden_units, task_hidden_units, feat_vocab,
                 embedding_size):
        super().__init__()
        self.embedding_layer = {feat: keras.layers.Embedding(vocab_size, embedding_size) for feat, vocab_size in
                                feat_vocab.items()}
        self.num_tasks = num_tasks
        self.experts = [MLP(units=expert_hidden_units, last_activation='relu') for _ in range(num_experts)]
        self.gates = [layers.Dense(num_experts, activation='softmax', use_bias=True) for _ in range(num_tasks)]
        self.task_towers = [MLP(units=task_hidden_units, last_activation='sigmoid') for _ in range(num_tasks)]

    def compute_embedding(self, inputs):
        embedding = [self.embedding_layer[feat](inputs[feat]) for feat in inputs]
        embedding = tf.concat(embedding, axis=-1)
        embedding = tf.squeeze(embedding, axis=1)
        return embedding

    def call(self, inputs, training=None, mask=None):
        inputs = self.compute_embedding(inputs)
        experts_outputs = []
        for expert in self.experts:
            expert_output = expert(inputs)
            expert_output = tf.expand_dims(expert_output, axis=0)
            experts_outputs.append(expert_output)
        experts_outputs = tf.concat(experts_outputs, axis=0)  # num_experts,None,expert_hidden_units[-1]
        experts_outputs = tf.transpose(experts_outputs, perm=(1, 0, 2))  # None,num_experts,expert_hidden_units[-1]
        outputs = []
        for i in range(self.num_tasks):
            gate = self.gates[i]
            task_tower = self.task_towers[i]
            gate_weights = gate(inputs)  # None,num_experts
            gate_weights = tf.expand_dims(gate_weights, axis=1)  # None,1,num_experts
            weighted_outputs = tf.matmul(gate_weights, experts_outputs)  # None,1,expert_hidden_units[-1]
            weighted_outputs = tf.squeeze(weighted_outputs, axis=1)
            task_output = task_tower(weighted_outputs)
            outputs.append(task_output)
        outputs[1] = outputs[0] * outputs[1]
        outputs = tf.concat(outputs, axis=1)
        return outputs

    def compute_cvr(self, inputs):
        inputs = self.compute_embedding(inputs)
        experts_outputs = []
        for expert in self.experts:
            expert_output = expert(inputs)
            expert_output = tf.expand_dims(expert_output, axis=0)
            experts_outputs.append(expert_output)
        experts_outputs = tf.concat(experts_outputs, axis=0)  # num_experts,None,expert_hidden_units[-1]
        experts_outputs = tf.transpose(experts_outputs, perm=(1, 0, 2))  # None,num_experts,expert_hidden_units[-1]
        outputs = []
        for i in range(self.num_tasks):
            gate = self.gates[i]
            task_tower = self.task_towers[i]
            gate_weights = gate(inputs)  # None,num_experts
            gate_weights = tf.expand_dims(gate_weights, axis=1)  # None,1,num_experts
            weighted_outputs = tf.matmul(gate_weights, experts_outputs)  # None,1,expert_hidden_units[-1]
            weighted_outputs = tf.squeeze(weighted_outputs, axis=1)
            task_output = task_tower(weighted_outputs)
            outputs.append(task_output)
        return outputs[1]

    def compute_ctr(self, inputs):
        inputs = self.compute_embedding(inputs)
        experts_outputs = []
        for expert in self.experts:
            expert_output = expert(inputs)
            expert_output = tf.expand_dims(expert_output, axis=0)
            experts_outputs.append(expert_output)
        experts_outputs = tf.concat(experts_outputs, axis=0)  # num_experts,None,expert_hidden_units[-1]
        experts_outputs = tf.transpose(experts_outputs, perm=(1, 0, 2))  # None,num_experts,expert_hidden_units[-1]
        outputs = []
        for i in range(self.num_tasks):
            gate = self.gates[i]
            task_tower = self.task_towers[i]
            gate_weights = gate(inputs)  # None,num_experts
            gate_weights = tf.expand_dims(gate_weights, axis=1)  # None,1,num_experts
            weighted_outputs = tf.matmul(gate_weights, experts_outputs)  # None,1,expert_hidden_units[-1]
            weighted_outputs = tf.squeeze(weighted_outputs, axis=1)
            task_output = task_tower(weighted_outputs)
            outputs.append(task_output)
        return outputs[0]

    def compute_ctcvr(self, inputs):
        inputs = self.compute_embedding(inputs)
        experts_outputs = []
        for expert in self.experts:
            expert_output = expert(inputs)
            expert_output = tf.expand_dims(expert_output, axis=0)
            experts_outputs.append(expert_output)
        experts_outputs = tf.concat(experts_outputs, axis=0)  # num_experts,None,expert_hidden_units[-1]
        experts_outputs = tf.transpose(experts_outputs, perm=(1, 0, 2))  # None,num_experts,expert_hidden_units[-1]
        outputs = []
        for i in range(self.num_tasks):
            gate = self.gates[i]
            task_tower = self.task_towers[i]
            gate_weights = gate(inputs)  # None,num_experts
            gate_weights = tf.expand_dims(gate_weights, axis=1)  # None,1,num_experts
            weighted_outputs = tf.matmul(gate_weights, experts_outputs)  # None,1,expert_hidden_units[-1]
            weighted_outputs = tf.squeeze(weighted_outputs, axis=1)
            task_output = task_tower(weighted_outputs)
            outputs.append(task_output)
        return outputs[0] * outputs[1]
