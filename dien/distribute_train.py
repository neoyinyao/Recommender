import argparse
import os
import random
import time

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from data_loader import example_generator, item_vocab, cat_vocab
from model import BaseModel, DIN, DIEN


class Train(object):
    def __init__(self, strategy: tf.distribute.Strategy, epochs, model: keras.Model, train_global_batch_size,
                 model_type, test_steps):
        self.test_steps = test_steps
        self.train_global_batch_size = train_global_batch_size
        self.epochs = epochs
        self.model = model
        self.strategy = strategy
        self.optimizer = keras.optimizers.Adam()
        self.train_auc = keras.metrics.AUC()
        self.test_auc = keras.metrics.AUC()
        self.bce_loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction=keras.losses.Reduction.NONE)
        self.train_step = self.compute_train_step_fn(model_type)

    def compute_train_step_fn(self, model_type):
        @tf.function
        def train_step_dien(inputs):
            def compute_per_replica_loss_dien(labels, prob, auxiliary_loss):
                bce_loss = self.bce_loss_fn(labels, prob)
                per_replica_bce_loss = tf.reduce_sum(bce_loss) * (1. / self.train_global_batch_size)
                per_replica_auxiliary_loss = tf.reduce_sum(auxiliary_loss) * (1. / self.train_global_batch_size)
                per_replica_loss = per_replica_bce_loss + per_replica_auxiliary_loss
                return per_replica_loss

            feature, label = inputs
            with tf.GradientTape() as tape:
                pred = self.model(feature, training=True)
                prob, auxiliary_loss = pred
                loss = compute_per_replica_loss_dien(label, prob, auxiliary_loss)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_auc.update_state(label, prob)
            return loss

        @tf.function
        def train_step_base_and_din(inputs):
            def compute_per_replica_loss_din(labels, pred):
                bce_loss = self.bce_loss_fn(labels, pred)
                # per_replica_loss = tf.reduce_sum(bce_loss) * (1. / self.train_global_batch_size)
                per_replica_loss = tf.nn.compute_average_loss(bce_loss, global_batch_size=self.train_global_batch_size)
                return per_replica_loss

            feature, label = inputs
            with tf.GradientTape() as tape:
                pred = self.model(feature, training=True)
                loss = compute_per_replica_loss_din(label, pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_auc.update_state(label, pred)
            return loss

        if model_type == 'BASE' or model_type == 'DIN':
            return train_step_base_and_din
        elif model_type == 'DIEN':
            return train_step_dien

    def test_step(self, inputs):
        feature, label = inputs
        pred = self.model.compute_prob(feature)
        self.test_auc.update_state(label, pred)

    def custom_loops(self, train_dist_dataset, test_dist_dataset):
        @tf.function
        def test_epoch(ds):
            test_start = time.time()
            for inputs in ds:
                self.strategy.run(self.test_step, args=(inputs,))
            test_end = time.time()
            test_info = 'time {0} auc{1:.4f}'.format(test_end - test_start,
                                                     self.train_auc.result().numpy())
            return test_info

        def train_epoch(ds):
            train_start = time.time()
            tqdm_util = tqdm(enumerate(ds, start=1))
            step_info = 'step {0:d} total_loss{1:.4f} step_loss{2:.4f} train_auc{3:.4f}'
            total_loss = 0.
            for step, one_batch in tqdm_util:
                per_replica_loss = self.strategy.run(self.train_step, args=(one_batch,))
                step_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
                total_loss += step_loss
                tqdm_util.set_description(
                    step_info.format(step, total_loss / step, step_loss, self.train_auc.result().numpy()))
            train_end = time.time()
            train_info = 'time {0} loss {1:.4f} auc{2:.4f}'.format(train_end - train_start, total_loss / step,
                                                                   self.train_auc.result().numpy())
            return train_info

        for epoch in range(self.epochs):
            train_info = train_epoch(train_dist_dataset)
            test_info = test_epoch(test_dist_dataset)
            print('epoch {} train_info {} test_info{}'.format(epoch, train_info, test_info))
            self.train_auc.reset_states()
            self.test_auc.reset_states()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--gpu_memory_limit', type=int, default=4096)
    parser.add_argument('--model_type', type=str, default='DIEN')
    parser.add_argument('--history_max_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--train_global_batch_size', type=int, default=128,
                        help='train global batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='test batch size.')
    parser.add_argument('--test_steps', type=int, default=2048,
                        help='test every train_steps')
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        devices = gpus
    else:
        devices = tf.config.list_logical_devices('CPU')
    strategy = tf.distribute.MirroredStrategy(devices)
    model_type = args.model_type
    seed = args.seed
    random.seed(seed)
    tf.random.set_seed(seed)
    epochs = args.epochs
    train_global_batch_size = args.train_global_batch_size
    test_batch_size = args.test_batch_size
    item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size = len(item_vocab), 18, len(
        cat_vocab), 18
    mlp_units = [200, 80, 1]  # same as official
    history_max_length = args.history_max_length
    test_steps = args.test_steps
    train_file = 'data/local_train_splitByUser'
    test_file = 'data/local_test_splitByUser'
    if model_type == 'BASE' or model_type == 'DIN':
        output_shapes = ({'target_item': tf.TensorShape((1,)), 'target_cat': tf.TensorShape((1,)),
                          'pos_his_item': tf.TensorShape((history_max_length,)),
                          'pos_his_cat': tf.TensorShape((history_max_length,))}, tf.TensorShape((1,)))
        output_types = ({'target_item': tf.int32, 'target_cat': tf.int32,
                         'pos_his_item': tf.int32, 'pos_his_cat': tf.int32}, tf.float32)
    elif model_type == 'DIEN':
        output_shapes = ({'target_item': tf.TensorShape((1,)), 'target_cat': tf.TensorShape((1,)),
                          'pos_his_item': tf.TensorShape((history_max_length,)),
                          'pos_his_cat': tf.TensorShape((history_max_length,)),
                          'neg_his_item': tf.TensorShape((history_max_length,)),
                          'neg_his_cat': tf.TensorShape((history_max_length,))
                          }, tf.TensorShape((1,)))

        output_types = ({'target_item': tf.int32, 'target_cat': tf.int32,
                         'pos_his_item': tf.int32, 'pos_his_cat': tf.int32,
                         'neg_his_item': tf.int32, 'neg_his_cat': tf.int32}, tf.float32)
    train_dataset = tf.data.Dataset.from_generator(example_generator, output_types=output_types,
                                                   output_shapes=output_shapes,
                                                   args=(model_type, train_file, history_max_length))
    test_dataset = tf.data.Dataset.from_generator(example_generator, output_types=output_types,
                                                  output_shapes=output_shapes,
                                                  args=(model_type, test_file, history_max_length))
    prefetch_size = 10
    train_dataset = train_dataset.shuffle(train_global_batch_size * 10).batch(train_global_batch_size).prefetch(
        prefetch_size)
    test_dataset = test_dataset.batch(test_batch_size).prefetch(prefetch_size)
    with strategy.scope():
        if model_type == 'DIEN':
            interest_extract_gru_units, interest_evolve_gru_units = 36, 36
            model = DIEN(item_vocab_size=item_vocab_size, item_embedding_size=item_embedding_size,
                         cat_vocab_size=cat_vocab_size,
                         cat_embedding_size=cat_embedding_size,
                         interest_extract_gru_units=interest_extract_gru_units,
                         interest_evolve_gru_units=interest_evolve_gru_units, mlp_units=mlp_units)
        elif model_type == 'BASE':
            model = BaseModel(item_vocab_size=item_vocab_size, item_embedding_size=item_embedding_size,
                              cat_vocab_size=cat_vocab_size, cat_embedding_size=cat_embedding_size, mlp_units=mlp_units)
        elif model_type == 'DIN':
            model = DIN(item_vocab_size=item_vocab_size, item_embedding_size=item_embedding_size,
                        cat_vocab_size=cat_vocab_size, cat_embedding_size=cat_embedding_size, mlp_units=mlp_units)
        trainer = Train(strategy, epochs, model, train_global_batch_size=train_global_batch_size, model_type=model_type,
                        test_steps=test_steps)
        train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
        trainer.custom_loops(train_dist_dataset, test_dist_dataset)


if __name__ == '__main__':
    train()
"""
TODO: add dropout and batch normalization layer,fine tune model
"""
