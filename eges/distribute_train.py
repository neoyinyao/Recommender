import argparse
import os
import random

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from model import DeepWalk, GES, EGES


class Train(object):
    def __init__(self, strategy: tf.distribute.Strategy, model, epochs, train_global_batch_size, test_steps, ckpt_dir):
        self.strategy = strategy
        self.model = model
        self.epochs = epochs
        self.train_global_batch_size = train_global_batch_size
        self.test_steps = test_steps
        self.optimizer = keras.optimizers.Adam()
        self.test_auc = keras.metrics.AUC(num_thresholds=20000)
        self.ckpt = tf.train.Checkpoint(optimzier=self.optimizer, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_dir, max_to_keep=None, )

    def train_step(self, inputs):
        feature, labels = inputs[:-1], inputs[-1]
        with tf.GradientTape() as tape:
            logits = self.model(feature)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
            loss /= logits.shape[-1]
            loss = tf.nn.compute_average_loss(loss, global_batch_size=self.train_global_batch_size)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def test_step(self, inputs):
        query_embedding, pos_embedding, neg_embedding = self.model.evaluation(inputs)
        pos_score = tf.reduce_sum(query_embedding * pos_embedding, axis=-1)  # None,1
        neg_score = tf.reduce_sum(query_embedding * neg_embedding, axis=-1)  # None,1
        pos_score = tf.nn.sigmoid(pos_score)
        neg_score = tf.nn.sigmoid(neg_score)
        pos_label = tf.ones_like(pos_score)
        neg_label = tf.zeros_like(pos_score)
        self.test_auc.update_state(pos_label, pos_score)
        self.test_auc.update_state(neg_label, neg_score)

    def custom_loop(self, train_dataset, test_dataset):
        @tf.function
        def distribute_train_step(inputs):
            per_replica_loss = self.strategy.run(self.train_step, args=(inputs,))
            total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            return total_loss

        @tf.function
        def distribute_test(ds):
            for inputs in ds:
                self.strategy.run(self.test_step, args=(inputs,))

        def distribute_train_epoch(ds):
            tqdm_util = tqdm(enumerate(ds, start=1))
            train_step_info = 'train step {0:d} step_loss {1:.4f}'
            total_loss = 0.
            for step, inputs in tqdm_util:
                step_loss = distribute_train_step(inputs)
                total_loss += step_loss
                tqdm_util.set_description(train_step_info.format(step, float(step_loss.numpy())))
                if step % self.test_steps == 0:
                    distribute_test(test_dataset)
                    test_info = 'test auc {0:.4f}'
                    print(test_info.format(float(self.test_auc.result().numpy())))
                    self.ckpt_manager.save()
            distribute_test(test_dataset)

        for epoch in range(self.epochs):
            distribute_train_epoch(train_dataset)


def main():
    parser = argparse.ArgumentParser(description="EGES model train config")
    parser.add_argument('--gpus', type=str, default='0', help='Which gpu to use.')
    parser.add_argument('--set_memory_growth', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='BGE', help='BGE or GES or EGES')
    parser.add_argument('--embedding_size', type=int, default=8, help='embedding_size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs.')
    parser.add_argument('--train_global_batch_size', type=int, default=1024, help='train batch size.')
    parser.add_argument('--test_global_batch_size', type=int, default=2048, help='test batch size.')
    parser.add_argument('--test_steps', type=int, default=3000, help='test steps')
    parser.add_argument('--seed', type=int, default=4, help='random seed')
    args = parser.parse_args()
    SEED = args.seed
    random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    epochs = args.epochs
    test_steps = args.test_steps
    embedding_size = args.embedding_size
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if args.set_memory_growth:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        logical_gpus = tf.config.list_logical_devices('GPU')
        devices = logical_gpus
    else:
        cpus = tf.config.list_logical_devices('CPU')
        devices = cpus
    os.environ['DGLBACKEND'] = 'tensorflow'
    from util import build_train_util
    from data_loader import build_dataset  # because import dgl automatic load all visible gpus, so import here
    train_g, test_item_pairs, item2idx, idx2item, train_item2cat, train_item2brand, brand_vocab, cat_vocab = \
        build_train_util()
    random_walk_length, num_ns = 10, 5
    model_type = 'BGE'
    train_dataset = build_dataset(model_type, 'train', train_g, random_walk_length, num_ns, cat_vocab,
                                  brand_vocab, train_item2cat,
                                  train_item2brand, item2idx, idx2item, test_item_pairs)
    test_dataset = build_dataset(model_type, 'test', train_g, random_walk_length, num_ns, cat_vocab,
                                 brand_vocab, train_item2cat,
                                 train_item2brand, item2idx, idx2item, test_item_pairs)
    strategy = tf.distribute.MirroredStrategy(devices)

    train_global_batch_size = args.train_global_batch_size
    test_global_batch_size = args.test_global_batch_size
    shuffle_ratio = 100
    train_dataset = train_dataset.shuffle(shuffle_ratio * train_global_batch_size).batch(
        train_global_batch_size)
    test_dataset = test_dataset.batch(test_global_batch_size)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)  # distribute input
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)  # distribute input
    ckpt_dir = './training_checkpoints/{}'.format(model_type)
    with strategy.scope():
        if model_type == 'BGE':
            model = DeepWalk(len(item2idx), embedding_size)
        elif model_type == 'GES':
            model = GES(len(item2idx), len(cat_vocab) + 1, len(brand_vocab) + 1, embedding_size)
        elif model_type == 'EGES':
            num_side = 2
            model = EGES(len(item2idx), len(cat_vocab) + 1, len(brand_vocab) + 1, embedding_size, num_side + 1)
        else:
            raise ValueError('model_type not valid')
        trainer = Train(strategy, model, epochs, train_global_batch_size, test_steps, ckpt_dir)
        trainer.custom_loop(train_dist_dataset, test_dist_dataset)


if __name__ == '__main__':
    main()
