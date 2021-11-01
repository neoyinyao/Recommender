import os
import argparse
import random
import tensorflow as tf
from tqdm import tqdm
from data_loader import din_example_generator, dien_example_generator, item_id_voc, item_cat_voc
from base import BaseModel
from din import DIN
from dien import DIEN


def train_din_and_base(model, epochs, train_file, train_batch_size, test_file, test_batch_size, history_max_length):
    @tf.function
    def din_train_step(batch_data):
        labels, features = batch_data[0], batch_data[1:]
        with tf.GradientTape() as tape:
            pred = model(features)
            pred = tf.squeeze(pred, axis=1)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_auc.update_state(labels, pred)
        return loss

    @tf.function
    def din_test_step(batch_data):
        labels, features = batch_data[0], batch_data[1:]
        pred = model(features)
        test_auc.update_state(labels, pred)

    output_shapes = (
        tf.TensorShape(()), tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((history_max_length,)),
        tf.TensorShape((history_max_length,)), tf.TensorShape((history_max_length,)))
    output_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
    train_dataset = tf.data.Dataset.from_generator(din_example_generator, output_types=output_types,
                                                   output_shapes=output_shapes, args=(train_file, history_max_length))
    test_dataset = tf.data.Dataset.from_generator(din_example_generator, output_types=output_types,
                                                  output_shapes=output_shapes, args=(test_file, history_max_length))
    prefetch_size = 10
    train_dataset = train_dataset.shuffle(train_batch_size * 100).batch(train_batch_size).prefetch(prefetch_size)
    test_dataset = test_dataset.batch(test_batch_size).prefetch(prefetch_size)
    train_auc = tf.keras.metrics.AUC(num_thresholds=20000)
    test_auc = tf.keras.metrics.AUC(num_thresholds=20000)
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(1, epochs + 1):
        train_tqdm_util = tqdm(train_dataset)
        test_tqdm_util = tqdm(test_dataset)
        for step, batch_data in enumerate(train_tqdm_util, start=1):
            loss = din_train_step(batch_data)
            train_tqdm_util.set_description(
                'train epoch {0:d} step {1:d} loss {2:.4f} auc {3:.4f}'.format(epoch, step, float(loss.numpy()),
                                                                               float(train_auc.result().numpy())))
            if step % 100 == 0:
                for test_batch_data in test_tqdm_util:
                    din_test_step(test_batch_data)
                print('test epoch{0:d} step {1:d} auc{2:.4f}'.format(epoch, step, float(test_auc.result().numpy())))
                test_auc.reset_states()
        train_auc.reset_states()


def train_dien(model, epochs, train_file, train_batch_size, test_file, test_batch_size, history_max_length):
    def dien_loss_fn(labels, pred, auxiliary_loss, weight=1):
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, pred))
        total_loss = loss + weight * auxiliary_loss
        return total_loss

    @tf.function
    def dien_train_step(batch_data):
        labels, features = batch_data[0], batch_data[1:]
        with tf.GradientTape() as tape:
            pred, auxiliary_loss = model(features)
            pred = tf.squeeze(pred, axis=1)
            total_loss = dien_loss_fn(labels, pred, auxiliary_loss)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_auc.update_state(labels, pred)
        return total_loss, auxiliary_loss

    @tf.function
    def dien_test_step(batch_data):
        labels, features = batch_data[0], batch_data[1:]
        pred, _ = model(features)
        test_auc.update_state(labels, pred)

    output_shapes = (
        tf.TensorShape(()), tf.TensorShape((1,)), tf.TensorShape((1,)), tf.TensorShape((history_max_length,)),
        tf.TensorShape((history_max_length,)), tf.TensorShape((history_max_length,)),
        tf.TensorShape((history_max_length,)), tf.TensorShape((history_max_length,)))
    output_types = (tf.float32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.int32, tf.float32)
    train_dataset = tf.data.Dataset.from_generator(dien_example_generator, output_types=output_types,
                                                   output_shapes=output_shapes, args=(train_file, history_max_length))
    test_dataset = tf.data.Dataset.from_generator(dien_example_generator, output_types=output_types,
                                                  output_shapes=output_shapes, args=(test_file, history_max_length))
    prefetch_size = 10
    train_dataset = train_dataset.shuffle(train_batch_size * 100).batch(train_batch_size).prefetch(prefetch_size)
    test_dataset = test_dataset.batch(test_batch_size).prefetch(prefetch_size)
    test_auc = tf.keras.metrics.AUC(num_thresholds=20000)
    train_auc = tf.keras.metrics.AUC(num_thresholds=20000)
    optimizer = tf.keras.optimizers.Adam()
    for epoch in range(1, epochs + 1):
        train_tqdm_util = tqdm(train_dataset)
        test_tqdm_util = tqdm(test_dataset)
        for step, batch_data in enumerate(train_tqdm_util, start=1):
            total_loss, auxiliary_loss = dien_train_step(batch_data)
            train_tqdm_util.set_description(
                'train epoch {0:d} step {1:d} total_loss {2:.4f} auxiliary_loss {3:.4f} auc{4:.4f}'.format(
                    epoch, step, float(total_loss.numpy()), float(auxiliary_loss.numpy()),
                    float(train_auc.result().numpy())))
            if step % 100 == 0:
                for test_batch_data in test_tqdm_util:
                    dien_test_step(test_batch_data)
                print('test epoch{0:d} step {1:d} auc{2:.4f}'.format(epoch, step, float(test_auc.result().numpy())))
                test_auc.reset_states()
        train_auc.reset_states()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--gpu_memory_limit', type=int, default=4096)
    parser.add_argument('--model_type', type=str, default='DIEN')
    parser.add_argument('--history_max_length', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs.')
    parser.add_argument('--train_batch_size', type=int, default=512,
                        help='train batch size.')
    parser.add_argument('--test_batch_size', type=int, default=2048,
                        help='test batch size.')
    parser.add_argument('--seed', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate gpu_memory_limit of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=args.gpu_memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    model_type = args.model_type
    seed = args.seed
    random.seed(seed)
    tf.random.set_seed(seed)
    epochs = args.epochs
    train_batch_size = args.train_batch_size
    test_batch_size = args.test_batch_size
    item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size = len(item_id_voc), 18, len(
        item_cat_voc), 18
    fc_units = [200, 80, 1]
    history_max_length = args.history_max_length
    train_file = 'data/local_train_splitByUser'
    test_file = 'data/local_test_splitByUser'
    if model_type == 'DIEN':
        interest_extract_gru_units, interest_evolve_gru_units = 36, 36
        model = DIEN(item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size,
                     interest_extract_gru_units,
                     interest_evolve_gru_units, fc_units)
        train_dien(model, epochs, train_file, train_batch_size, test_file, test_batch_size, history_max_length)
    elif model_type == 'BASE':
        model = BaseModel(item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size, fc_units)
        train_din_and_base(model, epochs, train_file, train_batch_size, test_file, test_batch_size, history_max_length)
    elif model_type == 'DIN':
        model = DIN(item_vocab_size, item_embedding_size, cat_vocab_size, cat_embedding_size, fc_units,
                    history_max_length)
        train_din_and_base(model, epochs, train_file, train_batch_size, test_file, test_batch_size, history_max_length)


if __name__ == '__main__':
    train()
"""
TODO: add dropout and batch normalization layer,fine tune model
"""
