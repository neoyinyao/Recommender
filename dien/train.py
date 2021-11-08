import argparse
import os
import random

import tensorflow as tf
from tqdm import tqdm

from data_loader import example_generator, item_vocab, cat_vocab
from model import BaseModel, DIN, DIEN


def train(model, model_type, epochs, train_ds, test_ds):
    @tf.function
    def train_step_dien(one_batch):
        feature, label = one_batch
        with tf.GradientTape() as tape:
            pred, auxiliary_loss = model(feature, training=True)
            bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label, pred))
            auxiliary_loss = tf.reduce_mean(auxiliary_loss)
            total_loss = bce_loss + auxiliary_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_auc.update_state(label, pred)
        return total_loss, auxiliary_loss

    @tf.function
    def train_step_base_and_din(one_batch):
        feature, label = one_batch
        with tf.GradientTape() as tape:
            pred = model(feature, training=True)
            bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(label, pred))
        gradients = tape.gradient(bce_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_auc.update_state(label, pred)
        return bce_loss

    @tf.function
    def test_step(one_batch):
        feature, label = one_batch
        pred = model.compute_prob(feature)
        test_auc.update_state(label, pred)

    train_auc = tf.keras.metrics.AUC(num_thresholds=20000)
    test_auc = tf.keras.metrics.AUC(num_thresholds=20000)
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(1, epochs + 1):
        train_tqdm_util = tqdm(train_ds)
        for step, batch_data in enumerate(train_tqdm_util, start=1):
            if model_type == 'DIEN':
                step_info = 'train epoch {0:d} step {1:d} total_loss {2:.4f} auxiliary_loss {3:.4f} auc{4:.4f}'
                total_loss, auxiliary_loss = train_step_dien(batch_data)
                train_tqdm_util.set_description(
                    step_info.format(epoch, step, float(total_loss.numpy()), float(auxiliary_loss.numpy()),
                                     float(train_auc.result().numpy())))
            elif model_type == 'BASE' or model_type == 'DIN':
                step_info = 'train epoch {0:d} step {1:d} total_loss {2:.4f}  auc{3:.4f}'
                total_loss = train_step_base_and_din(batch_data)
                train_tqdm_util.set_description(
                    step_info.format(epoch, step, float(total_loss.numpy()), float(train_auc.result().numpy())))
            if step % 100 == 0:
                test_tqdm_util = tqdm(test_ds)
                for test_batch_data in test_tqdm_util:
                    test_step(test_batch_data)
                print('test epoch{0:d} step {1:d} auc{2:.4f}'.format(epoch, step, float(test_auc.result().numpy())))
                test_auc.reset_states()
        train_auc.reset_states()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--gpu_memory_limit', type=int, default=4096)
    parser.add_argument('--model_type', type=str, default='BASE')
    parser.add_argument('--history_max_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='train batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='test batch size.')
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    model_type = args.model_type
    seed = args.seed
    random.seed(seed)
    tf.random.set_seed(seed)
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size = len(item_vocab), 18, len(
        cat_vocab), 18
    mlp_units = [200, 80, 1]  # same as official
    history_max_length = args.history_max_length
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
    train_dataset = train_dataset.shuffle(train_batch_size * 10).batch(train_batch_size).prefetch(prefetch_size)
    test_dataset = test_dataset.batch(test_batch_size).prefetch(prefetch_size)
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
    train(model, model_type, epochs, train_dataset, test_dataset)


if __name__ == '__main__':
    main()
"""
TODO: add dropout and batch normalization layer,fine tune model
"""
